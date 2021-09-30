// opengl_3d.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include <iostream>
#include <GL/freeglut.h>

#define X 0
#define Y 1
#define Z 2

unsigned int num = 8;
double points[][3] = {	{1.0, 1.0, -1.0},
						{-1.0, 1.0, -1.0},
						{-1.0, -1.0, -1.0},
						{1.0, -1.0, -1.0},
						{1.0, 1.0, 1.0},
						{-1.0, 1.0, 1.0},
						{-1.0, -1.0, 1.0},
						{1.0, -1.0, 1.0} };

int lines[][2] = { {0, 1},
					{1, 2},
					{2, 3},
					{3, 0},
					{4, 5},
					{5, 6},
					{6, 7},
					{7, 4},
					{0, 4},
					{1, 5},
					{2, 6},
					{3, 7} };

double eye[3] = { 2.0, 1.0, 1.0 };
double center[3] = { 0.0, 0.0, 0.0 };
double up[3] = { 0.0, 0.0, 1.0 };

float color_red[3] = { 1.0, 0.0, 0.0 };
float color_green[3] = { 0.0, 1.0, 0.0 };
float color_blue[3] = { 0.0, 0.0, 1.0 };



double left = -2.0;
double right = 2.0;
double bottom = -2.0;
double top = 2.0;

int width = 768;
int height = 768;

void display(void);


void createWindow(char* title, int x, int y, int width, int height) {	
	glutInitWindowPosition(x, y);
	glutInitWindowSize(width, height);
	glutCreateWindow(title);
}

int main(int argc, char *argv[])
{
    std::cout << "Hello World!\n";
	glutInit(&argc, argv);
	char title[] = "Hell World";
	createWindow(title, 0, 0, width, height);

	glutDisplayFunc(display);
	glutMainLoop();
}

void initGL(void) {
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

void display(void) {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(left, right, bottom, top, -100.0, 100.0);
	glViewport(0, 0, width, height);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(eye[X], eye[Y], eye[Z], center[X], center[Y], center[Z], up[X], up[Y], up[Z]);
	glClear(GL_COLOR_BUFFER_BIT);
	glBegin(GL_LINES);
	for (int i = 0; i < 12; i++) {
		int j0 = lines[i][0];
		glVertex3dv(points[j0]);
		int j1 = lines[i][1];
		glVertex3dv(points[j1]);
	}
	glEnd();
	glFlush();
}



