#include <stdlib.h>
#include <time.h>

void sum(float *a, float* b, float* c, const int num) {
	for (int idx = 0; idx < num; idx++) {
		c[idx] = a[idx] + b[idx];
	}
}


void genData(float *ip, int size) {
	time_t t;
	srand((unsigned int) time(&t));
	for (int i = 0; i < size; i++) {
		ip[i] = (float)(rand() & 0xff) / 10.0f;
	}
	return;
}

int main(int argc, char** argv) {
	float* ha;
	float* hb;
	float* hc;

	int n = 1024;
	size_t bytes = n * sizeof(float);

	ha = (float*) malloc(bytes);
	hb = (float*) malloc(bytes);
	hc = (float*) malloc(bytes);

	genData(ha, n);
	genData(hb, n);
	sum(ha, hb, hc, n);

	free(ha);
	free(hb);
	free(hc);
}
