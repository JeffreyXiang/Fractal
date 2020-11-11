#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>

#include "cuda_runtime.h"
#include <cuda_gl_interop.h>  
#include "device_launch_parameters.h"
#include "kernel.h"

int2 size = { 1920, 1080 };                                     //窗口（图像）尺寸
double diagonal = sqrt(size.x * size.x + size.y * size.y);      //对角线

typedef struct
{
    double zoom;
    double xC;
    double yC;
} Zone;

std::vector<Zone> zones;                                        //可视区域栈
bool mode;                                                      //模式
int max_Iterations;                                             //最大迭代数
double2 JuliaPos;                                               //Julia集位置参数

uchar3 palette[256];                                            //调色板
uchar3* palette_d;                                              //GPU端调色板数据指针

GLuint PBO;                                                     //OpenGL像素缓冲对象
cudaGraphicsResource* resource;                                 //CUDA图形资源对象

bool mousePressed = false;                                      //鼠标状态flag
double2 start, end;                                             //鼠标控制点位置

void refreshImage(GLFWwindow* window);

//窗口大小改变回调
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    //更新数据
    size.x = width;
    size.y = height;
    diagonal = sqrt(width * width + height * height);

    //更新视口和像素缓冲对象
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, size.x, size.y, 0, -1, 1);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size.x * size.y * 4, NULL, GL_DYNAMIC_DRAW);

    refreshImage(window);
}

//鼠标点击回调
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS && !mousePressed)
        {
            //左键按下：初始化控制点
            mousePressed = true;
            glfwGetCursorPos(window, &start.x, &start.y);
            end = start;
        }
        else if (action == GLFW_RELEASE && mousePressed)
        {
            //左键抬起：计算新视口，刷新
            Zone temp = zones.back();
            mousePressed = false;
            double xC = temp.xC + (start.x - size.x / 2.0) / size.x * temp.zoom;
            double yC = temp.yC - (start.y - size.y / 2.0) / size.x * temp.zoom;
            double zoom = temp.zoom * sqrt(pow(start.x - end.x, 2) + pow(start.y - end.y, 2)) * 2 / diagonal;
            zones.push_back({ zoom, xC, yC });
            refreshImage(window);
        }
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
    {
        if (!mousePressed && zones.size() > 1)
        {
            //右键：回退视口
            zones.pop_back();
            refreshImage(window);
        }
        else
        {
            //右键：取消控制
            mousePressed = false;
            refreshImage(window);
        }
    }
}

//鼠标移动回调
static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (mousePressed)
    {
        //更新控制点
        end.x = xpos;
        end.y = ypos;

        //计算示意框
        double len = sqrt(pow(start.x - end.x, 2) + pow(start.y - end.y, 2));
        double x = len * size.x / diagonal;
        double y = len * size.y / diagonal;

        //绘制示意框
        glDrawPixels(size.x, size.y, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBegin(GL_LINE_LOOP);
        glColor4f(1, 0, 0, 1);
        glVertex2d(start.x + x, start.y + y);
        glVertex2d(start.x + x, start.y - y);
        glVertex2d(start.x - x, start.y - y);
        glVertex2d(start.x - x, start.y + y);
        glEnd();
        glFlush();
        glfwSwapBuffers(window);
    }
}

//ESC退出
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

//初始化调色板
void initPalette()
{
	for (int i = 0; i < 256; i++)
	{
        palette[i % 256].x = i % 256 / 64 == 0 ? i : i % 256 / 64 == 1 ? -128 + 3 * i : i % 256 / 64 == 2 ? 383 - i : 768 - 3 * i;
        palette[i % 256].y = i % 256 / 32 == 0 ? i : i % 256 / 32 == 1 || i % 256 / 32 == 2 ? -64 + 3 * i : i % 256 / 32 == 3 ? 128 + i : i % 256 / 32 == 4 ? 383 - i : i % 256 / 32 == 5 || i % 256 / 32 == 6 ? 703 - 3 * i : 255 - i;
        palette[i % 256].z = i % 256 / 64 == 0 ? 3 * i : i % 256 / 64 == 1 ? 128 + i : i % 256 / 64 == 2 ? -4 * i - 1 : 0;
	}
    cudaMalloc((void**)&palette_d, 256 * sizeof(uchar3));
    cudaMemcpy(palette_d, palette, 256 * sizeof(uchar3), cudaMemcpyHostToDevice);
}

//刷新图像
void refreshImage(GLFWwindow* window)
{
    //CUDA映射GPU端图形资源
    uchar4* img_d;
    cudaGraphicsGLRegisterBuffer(&resource, PBO, cudaGraphicsMapFlagsNone);
    cudaGraphicsMapResources(1, &resource, NULL);
    cudaGraphicsResourceGetMappedPointer((void**)&img_d, NULL, resource);

    //计算可视区域
    Zone z = zones.back();
    double4 zone;
    zone.x = z.xC - z.zoom / 2;
    zone.z = z.xC + z.zoom / 2;
    zone.y = z.yC - z.zoom / 2 * size.y / size.x;
    zone.w  = z.yC + z.zoom / 2 * size.y / size.x;

    //CUDA计算
    if (mode)
        Mandelbrot(img_d, palette_d, size, zone, max_Iterations);
    else
        Julia(img_d, palette_d, size, JuliaPos, zone, max_Iterations);

    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "addKernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        exit(-1);
    }

    //CUDA图形资源解映射
    cudaGraphicsUnmapResources(1, &resource, NULL);

    //OpenGL绘制像素
    glDrawPixels(size.x, size.y, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glfwSwapBuffers(window);
}

int main()
{
    //初始区域
    zones.push_back({ 4, 0, 0 });

    std::cout << "Fractal with OpenGL and CUDA\n";

    std::cout << "Select Mode (1 for Mandelbrot; 0 for Julia): ";
    std::cin >> mode;

    std::cout << "Max Iteration: ";
    std::cin >> max_Iterations;

    if (!mode)
    {
        std::cout << "Julia Pos: ";
        std::cin >> JuliaPos.x >> JuliaPos.y;
    }

    //初始化OpenGL到兼容模式
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    //创建窗口
    GLFWwindow* window = glfwCreateWindow(size.x, size.y, "Fractal", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    //初始化OpenGL API
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    //初始化视口
    glViewport(0, 0, size.x, size.y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, size.x, size.y, 0, -1, 1);

    //绑定事件处理回调
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    //创建像素缓冲区
    glGenBuffers(1, &PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size.x * size.y * 4, NULL, GL_DYNAMIC_DRAW);

    //初始化调色盘
    initPalette();

    //刷新图像
    refreshImage(window);

    //循环处理事件
    while (!glfwWindowShouldClose(window))
    {
        processInput(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
