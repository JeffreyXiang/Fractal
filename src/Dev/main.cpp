#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>

#include "cuda_runtime.h"
#include <cuda_gl_interop.h>  
#include "device_launch_parameters.h"
#include "kernel.h"

int2 size = { 1920, 1080 };                                     //���ڣ�ͼ�񣩳ߴ�
double diagonal = sqrt(size.x * size.x + size.y * size.y);      //�Խ���

typedef struct
{
    double zoom;
    double xC;
    double yC;
} Zone;

std::vector<Zone> zones;                                        //��������ջ
bool mode;                                                      //ģʽ
int max_Iterations;                                             //��������
double2 JuliaPos;                                               //Julia��λ�ò���

uchar3 palette[256];                                            //��ɫ��
uchar3* palette_d;                                              //GPU�˵�ɫ������ָ��

GLuint PBO;                                                     //OpenGL���ػ������
cudaGraphicsResource* resource;                                 //CUDAͼ����Դ����

bool mousePressed = false;                                      //���״̬flag
double2 start, end;                                             //�����Ƶ�λ��

void refreshImage(GLFWwindow* window);

//���ڴ�С�ı�ص�
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    //��������
    size.x = width;
    size.y = height;
    diagonal = sqrt(width * width + height * height);

    //�����ӿں����ػ������
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, size.x, size.y, 0, -1, 1);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size.x * size.y * 4, NULL, GL_DYNAMIC_DRAW);

    refreshImage(window);
}

//������ص�
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS && !mousePressed)
        {
            //������£���ʼ�����Ƶ�
            mousePressed = true;
            glfwGetCursorPos(window, &start.x, &start.y);
            end = start;
        }
        else if (action == GLFW_RELEASE && mousePressed)
        {
            //���̧�𣺼������ӿڣ�ˢ��
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
            //�Ҽ��������ӿ�
            zones.pop_back();
            refreshImage(window);
        }
        else
        {
            //�Ҽ���ȡ������
            mousePressed = false;
            refreshImage(window);
        }
    }
}

//����ƶ��ص�
static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (mousePressed)
    {
        //���¿��Ƶ�
        end.x = xpos;
        end.y = ypos;

        //����ʾ���
        double len = sqrt(pow(start.x - end.x, 2) + pow(start.y - end.y, 2));
        double x = len * size.x / diagonal;
        double y = len * size.y / diagonal;

        //����ʾ���
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

//ESC�˳�
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

//��ʼ����ɫ��
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

//ˢ��ͼ��
void refreshImage(GLFWwindow* window)
{
    //CUDAӳ��GPU��ͼ����Դ
    uchar4* img_d;
    cudaGraphicsGLRegisterBuffer(&resource, PBO, cudaGraphicsMapFlagsNone);
    cudaGraphicsMapResources(1, &resource, NULL);
    cudaGraphicsResourceGetMappedPointer((void**)&img_d, NULL, resource);

    //�����������
    Zone z = zones.back();
    double4 zone;
    zone.x = z.xC - z.zoom / 2;
    zone.z = z.xC + z.zoom / 2;
    zone.y = z.yC - z.zoom / 2 * size.y / size.x;
    zone.w  = z.yC + z.zoom / 2 * size.y / size.x;

    //CUDA����
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

    //CUDAͼ����Դ��ӳ��
    cudaGraphicsUnmapResources(1, &resource, NULL);

    //OpenGL��������
    glDrawPixels(size.x, size.y, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glfwSwapBuffers(window);
}

int main()
{
    //��ʼ����
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

    //��ʼ��OpenGL������ģʽ
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    //��������
    GLFWwindow* window = glfwCreateWindow(size.x, size.y, "Fractal", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    //��ʼ��OpenGL API
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    //��ʼ���ӿ�
    glViewport(0, 0, size.x, size.y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, size.x, size.y, 0, -1, 1);

    //���¼�����ص�
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    //�������ػ�����
    glGenBuffers(1, &PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size.x * size.y * 4, NULL, GL_DYNAMIC_DRAW);

    //��ʼ����ɫ��
    initPalette();

    //ˢ��ͼ��
    refreshImage(window);

    //ѭ�������¼�
    while (!glfwWindowShouldClose(window))
    {
        processInput(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
