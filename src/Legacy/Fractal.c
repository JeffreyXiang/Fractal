#include<stdio.h>

//复数
struct complex
{
    double Real;
    double Imag;
};

//颜色
struct BGR
{
    unsigned char B;
    unsigned char G;
    unsigned char R;
};

int Width, Height, max_Iterations, mode;
double X_max, X_min, Y_max, Y_min, Delta, xC, yC;
struct BGR IMG[2000 * 3000], Color[257];

//生成颜色序列
void Build_color_array()
{
    int i;
    /*1*/
    /*for (i=1;i<=48;i++)
    {
        Color[(i+23)%48+1].B=i%48/8==5?32*i:i%48/16==0?255:i%48/8==2?255-32*i:0;
        Color[(i+23)%48+1].G=i%48/8==3?32*i:i%48/16==2?255:i%48/8==0?255-32*i:0;
        Color[(i+23)%48+1].R=i%48/8==1?32*i:i%48/16==1?255:i%48/8==4?255-32*i:0;
    }*/
    /*2*/
    for (i = 1;i <= 256;i++)
    {
        Color[(i) % 256 + 1].B = i % 256 / 64 == 0 ? 3 * i : i % 256 / 64 == 1 ? 128 + i : i % 256 / 64 == 2 ? -4 * i - 1 : 0;
        Color[(i) % 256 + 1].G = i % 256 / 32 == 0 ? i : i % 256 / 32 == 1 || i % 256 / 32 == 2 ? -64 + 3 * i : i % 256 / 32 == 3 ? 128 + i : i % 256 / 32 == 4 ? 383 - i : i % 256 / 32 == 5 || i % 256 / 32 == 6 ? 703 - 3 * i : 255 - i;
        Color[(i) % 256 + 1].R = i % 256 / 64 == 0 ? i : i % 256 / 64 == 1 ? -128 + 3 * i : i % 256 / 64 == 2 ? 383 - i : 768 - 3 * i;
    }
    Color[0].B = Color[0].G = Color[0].R = 0;
}

struct complex plus(struct complex A, struct complex B)
{
    struct complex C;
    C.Real = A.Real + B.Real;
    C.Imag = A.Imag + B.Imag;
    return C;
}

struct complex multi(struct complex A, struct complex B)
{
    struct complex C;
    C.Real = A.Real * B.Real - A.Imag * B.Imag;
    C.Imag = A.Real * B.Imag + A.Imag * B.Real;
    return C;
}

struct complex power(struct complex A, int n)
{
    struct complex C = A;
    int i;
    for (i = 2; i <= n; i++)
        C = multi(C, A);
    return C;
}

//迭代函数
struct complex f(struct complex Z, struct complex C)
{
    return plus(power(Z, 2), C);
}

//绘制图片
void draw_img()
{
    struct complex C, Z;
    double Delta_X = (X_max - X_min) / Width;
    double Delta_Y = (Y_max - Y_min) / Height;

    double a, b;
    if (!mode)
    {
        printf("Julia Pos: ");
        scanf("%lf%lf", &a, &b);
    }

    int r, c, i, s = 0;
    system("cls");
    printf("Processing ");
    for (i = 1; i <= 100; i++) putchar(' ');
    printf("|%2d%%", s);
    for (r = 1; r <= Height; r++)
    {
        for (c = 1; c <= Width; c++)
        {
            if (mode)
            {
                C.Real = X_min + c * Delta_X;
                C.Imag = Y_min + r * Delta_Y;
                Z.Real = Z.Imag = 0;
            }
            else
            {
                C.Real = a;
                C.Imag = b;
                Z.Real = X_min + c * Delta_X;
                Z.Imag = Y_min + r * Delta_Y;
            }
            for (i = 1; i <= max_Iterations; i++)
            {
                Z = f(Z, C);
                if (Z.Real * Z.Real + Z.Imag * Z.Imag > 4.0)
                {
                    IMG[(r - 1) * Width + c - 1] = Color[i % 256 + 1];
                    break;
                }
            }
        }
        if (r == (int)((Height / 100.0) * (int)(r / (Height / 100.0))) + 1)
        {
            s++;
            system("cls");
            printf("Processing ");
            for (i = 1;i <= s;i++) putchar('.');
            for (i = s + 1;i <= 100;i++) putchar(' ');
            printf("|%2d%%", s);
        }
    }
}

//输出到BMP
void Save_to_Bmp(const char* filename)
{
    unsigned int size = Width * Height * 3 + 54;
    unsigned short head[] = {
        0x4D42,size % 0x10000,size / 0x10000,0,0,0x36,0,0x28,
        0,Width % 0x10000,Width / 0x10000,Height % 0x10000,Height / 0x10000,0x10,0x18,0,
        0,0,0,0,0,0,0,0,0,0,0
    };
    printf("\nExporting...\n");
    FILE* fp = fopen(filename, "wb");
    if (!fp)
    {
        printf("ERROR: cannot open file.\n");
        return;
    }
    fwrite(head, 1, sizeof(head), fp);
    fwrite(IMG, 1, size, fp);
    fclose(fp);
}

main()
{
    printf("Fractal\n");

    printf("Select Mode (1 for Mandelbrot; 0 for Julia): ");
    scanf("%d", &mode);

    printf("Size: ");
    scanf("%d %d", &Width, &Height);

    printf("Zone Width: ");
    scanf("%lf", &Delta);

    printf("Center: ");
    scanf("%lf%lf", &xC, &yC);

    printf("Max Iteration: ");
    scanf("%d", &max_Iterations);

    X_max = xC + Delta / 2;
    X_min = xC - Delta / 2;
    Y_max = yC + Delta / 2 * Height / Width;
    Y_min = yC - Delta / 2 * Height / Width;

    Build_color_array();
    draw_img();
    Save_to_Bmp("../data/Picture.bmp");
    printf("Done.\n");
    getchar();
    while (getchar() != 10);
}
//X_max=-0.6717536270481832370389902348813046+Width*Delta/1000;
    //X_min=-0.6717536270481832370389902348813046-Width*Delta/1000;
    //Y_max=0.4609205425545165168733304746281938+Height*Delta/1000;
    //Y_min=0.4609205425545165168733304746281938-Height*Delta/1000;
