// gcc -shared -W -o lattice.so -fPIC lattice.c  pcg_basic.c

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

#include "pcg_basic.h"



void _construct_square_neighbor_table(int* table ,int Nx,int Ny){

    pcg32_srandom(42u, 54u); 

    const int connectivity = 4;
    int lattice_index;
    int u;


    for (int i=0;i<Nx;i++){
        for (int j=0;j<Ny; j++){
            
            lattice_index = i*Ny+j;
            u = lattice_index*connectivity;
            // up
            if (i==0){

                table[u+0] = (Nx-1)*Ny+j;
            }   
            else{
                table[u+0] = lattice_index-Nx;
            }
            // right
            if (j==Ny-1){
                table[u+1] = lattice_index-Ny+1;
         
            }   
            else{
                table[u+1] = lattice_index+1;
            }
             // down
            if (i==Nx-1){

                table[u+2] = j;
            }   
            else{
                table[u+2] = lattice_index+Nx;
            }
            // left
            if (j==0){

                table[u+3] = lattice_index+Ny-1;
            }   
            else{
                table[u+3] = lattice_index-1;

                // printf("Here %i\n", index);
            }

            lattice_index +=1;
        }

    }


}

void permutation(int* array, int size){
    int j;
    int copy;

    for (int i=0; i<size;i++){
        j = (int)pcg32_boundedrand(size);
        copy = array[i];
        array[i] = array[j];
        array[j] = copy;
    }
}
double uniform()
{
    return (double)pcg32_boundedrand(RAND_MAX) / (double)RAND_MAX ;
}

void _move( 
    int coordination,
    int nparticles, 
    int* table,
    int * orientation, 
    int* occupancy, 
    int* location,
    double tumble_probability,
    int repeat){

    int indices[nparticles];

    for (int i = 0; i < nparticles; ++i) indices[i] = i;

    for (int k=0; k<repeat; k++){
        int p,o,oo,site,attempt;
        double r;
        permutation(indices, nparticles);
        

        for (int i = 0; i < nparticles; ++i)
        {   
            // select a particle
            p = indices[i];
            // get its site
            site = location[p];
            // get its orientation
            o = orientation[p];
            // get destination site
            attempt = table[site*coordination+o];

            if (occupancy[attempt]==0){
                occupancy[site] = 0;
                occupancy[attempt] = 1;
                location[p] = attempt;
            }
            r= uniform();
            // printf("%g\n",r);
            if (r<tumble_probability){
                oo = (int)pcg32_boundedrand(coordination); 
                // printf("old %d new %d\n",o,oo );
                orientation[p] = oo;
            }

        }
    }
}