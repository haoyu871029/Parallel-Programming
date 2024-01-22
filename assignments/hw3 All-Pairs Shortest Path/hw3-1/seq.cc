#include <stdio.h>
#include <stdlib.h>

const int INF = ((1 << 30) - 1);
const int V = 50010;
int n, m;
static int Dist[V][V];

void input(char* infile) {

    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    /* default */
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    /* initialize */
    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }

    fclose(file);
} //Dist array ok.

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            if (Dist[i][j] >= INF) 
                Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

void FW(){
    for (int k=0; k<n; ++k){
        for (int i=0; i<n; ++i){
            for (int j=0; j<n; ++j){
                if (Dist[i][k] != INF && Dist[i][j] > Dist[i][k] + Dist[k][j])
                    Dist[i][j] = Dist[i][k] + Dist[k][j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    input(argv[1]);
    FW();
    output(argv[2]);
    return 0;
}

//copy to hw3-1.cc
//compile: "g++ -O3 -o hw3-1 hw3-1.cc" or "make"
//execute: ./hw3-1 /home/pp23/share/hw3-1/cases/c01.1 c01.1.out