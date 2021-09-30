// opengl_2d.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//
//#include "stdafx.h"

#include <iostream>
#include <GL/freeglut.h>

#define X 0
#define Y 1
#define Z 2

unsigned int num = 5;
double point[][3] = {	{1.3, 1.3, 0.0},
						{0.3, 1.3, 0.0},
						{0.3, 0.3, 0.0},
						{1.3, 0.3, 0.0},
						{0.8, 0.8, 0.0} };

int triangles[][3] = { {0, 1, 4}, {1, 2, 4}, {2, 3, 4} };

float color_red[3] = { 1.0, 0.0, 0.0 };
float color_green[3] = { 0.0, 1.0, 0.0 };
float color_blue[3] = { 0.0, 0.0, 1.0 };


void displayLine(void);
void displayPolygon(void);
void displayTriangle(void);

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
	createWindow(title, 0, 0, 400, 400);

	glutDisplayFunc(displayTriangle);
	glutMainLoop();
}

void initGL(void) {
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

void displayLine(void) {
	glClear(GL_COLOR_BUFFER_BIT);
	glBegin(GL_LINE_LOOP);
	for (int i = 0; i < num; i++) {
		glVertex3d(point[i][X], point[i][Y], point[i][Z]);
	}
	glEnd();
	glFlush();
}

void displayPolygon(void) {
	glClear(GL_COLOR_BUFFER_BIT);
	glBegin(GL_POLYGON);
	for (int i = 0; i < num; i++) {
		glVertex3d(point[i][X], point[i][Y], point[i][Z]);
	}
	glEnd();
	glFlush();
}

void displayTriangle(void) {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-2.0, 2.0, -2.0, 2.0, -100.0, 100.0);
	glViewport(0, 0, 400, 400);
	glClear(GL_COLOR_BUFFER_BIT);
	glBegin(GL_TRIANGLES);
	for (int i = 0; i < 3; i++) {
		int* triangle = triangles[i];
		for (int j = 0; j < 3; j++) {
			auto index = triangle[j];
			glColor3fv(color_red);
			glVertex3dv(point[index]);
		}
	}
	glEnd();
	glFlush();
}

