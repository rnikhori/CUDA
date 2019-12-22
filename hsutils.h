#include<cuda.h>
#include<iostream>
#include<curand.h>
#include<malloc.h>
#include<stdlib.h>
using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), _FILE, __LINE_); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error_) printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error_));}
void prnt(float * arr, int n){

	for(int i=0; i<n;i++){
		printf("%f ",arr[i] );
	}
	printf("\n");
}
void prnt(float ** arr, int m, int n ){
	for(int j=0; j<m; j++){
		for(int i=0 ;i< n;i++){
			printf("%f ",arr[j][i] );
		}
		printf("\n");
	}
}

float **alloc2d(int m,int n, float *prev =NULL){
	float * arr = (float*) malloc(sizeof(float )*m*n);
	if (prev != NULL){
		memcpy(arr, prev, sizeof(float)*m*n);
	}
	float *brr = (float* ) malloc(sizeof(float*)*m);
	for(int i=0 ; i<m; i++){
		brr[i] = &arr[n*i];
	}
	return brr ;
}
_global_ void objectiveFn(float *res, float *harmonics, int dim,
		int pitch, int population, float offset){

//	cudaMalloc(&res, population);
	int index = blockIdx.x*blockDim.x*pitch+ threadIdx.x*pitch;
	int rindex = (int) index/pitch;
	if ( rindex>=population ){
		return ;
	}

	res[rindex] = 0 ;
	for(int i=0; i<dim;i++ ){
		res[rindex] += (harmonics[index+i]+offset)*(harmonics[index+i]+offset);
	}
}
_global_ void scale_vector(float *vec, int findex,int low, int high , int dim){

	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid >= dim){
			return;
	}
	vec[findex+tid] = low+vec[findex+tid]*(high-low);
}
_global_ void inrange(float *pop, size_t pitch, int low, int high, int population, int dim){
	/**
			pop: population array len:(population x dimensions)
			pitch: pitch for the memeory
			low bound of search space
			high er bound of search space
			**/
	int index = threadIdx.x*pitch+ (blockIdx.x*blockDim.x*pitch);
	if(index/pitch>= population){
		return ;
	}
	if (dim>1000){
		scale_vector<<<(int )dim/200+1,200>>>( pop, index, low, high, dim);
	}
	else{
		scale_vector<<<1,dim>>>( pop, index, low, high, dim);
	}
}
float* gen_random(curandGenerator_t gen, int row, int col, size_t *pitch,int low,
int high){

	float *arr;
	cudaMallocPitch(&arr, pitch, sizeof(float)*col,sizeof(float)*row);
	curandGenerateUniform(gen, arr, *pitch*row);
	if(col < 1000){

		inrange<<<1, row>>>(arr, *pitch, low, high, row, col);
	}
	else{
		inrange<<<(int )row/200+1, 200>>>(arr, *pitch, low, high, row, col);
	}
	return arr;
}
_global_ void swap(float*a , float*b, int srcIndex, int destIndex, int dim){

	int id = threadIdx.x + blockDim.x*blockIdx.x;
	if(id>=dim ){
		return ;
	}
	float temp ;
	temp = a[id+srcIndex];
	a[id+srcIndex] = b[id+destIndex];
	b[id+destIndex] = temp;
}
_global_ void sorted(float *res, float *harmonics, float *obj, float *sortedObj, int population , int dim, int pitch){



	int myPos= (blockIdx.x*blockDim.x)+threadIdx.x;
	//obj[0]=1;
	if(myPos >= population){
		return ;
	}


	//find resultant positions
	float myobjective =  obj[myPos];
	int countSmaller=0;

	for (int i=0 ; i< population; i++ ){
		if(myobjective>= obj[i] and i!=myPos){
			if(i< myPos && myobjective== obj[i]){
				continue;
			}
			countSmaller++;
		}
	}
	//for duplicate entries
	int srcIndex = myPos*pitch;
	int destIndex = countSmaller*pitch;
	sortedObj[countSmaller] = obj[myPos];

	if (dim>1000){
		swap<<<(int)(dim/100)+1,100>>>(res, harmonics, destIndex, srcIndex, dim);
	}
	else{
		swap<<<1, dim>>>(res, harmonics,  destIndex,srcIndex, dim);
	}


}
_global_ void harmonic_update(float harmonics, float *bests,float noise,
								int hindex, int bindex, float brange, int dim){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id>= dim){
		return ;
	}
	harmonics[hindex+id] = bests[bindex+id]+ noise[hindex+id]*brange;
}
_global_ void update_harmonics( float *harmonics,float *bests,
		float *noise, int *randomRecs, float brange, int population,
		int dim, size_t pitch ){

	//generate N(dim) random row and col indexes scale them
	int index = threadIdx.x+blockIdx.x*blockDim.x;
	if (index>=population){
		return;
	}
	int recIndex = randomRecs[index];
	int hindex = pitch*index;
	int bindex = recIndex*pitch;

	if(dim>1000){
		int blocks = dim%100 ==0 ? (int)dim/100 : ((int)dim/100)+1;
		harmonic_update<<<blocks, 100>>>(harmonics, bests, noise, hindex, bindex, brange,dim);
	}
	else{
		harmonic_update<<<1, dim>>>( harmonics, bests, noise, hindex, bindex, brange, dim);
	}
}
_global_ void cast_to_int(int *res, float *arr,int rows, int maxIndex){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id>= rows){return ;}
	res[id] = ((int) (arr[id]*maxIndex));
	if (res[id]<0){
			res[id] =0;
	}
}
int *gen_random_indexes(curandGenerator_t gen, int row, int maxIndex){

	float *arr;
	cudaMalloc(&arr, sizeof(float)*row);
	curandGenerateUniform(gen, arr, row);
	int* res;
	cudaMalloc(&res, sizeof(int)*row);

	if(row< 1000){
		cast_to_int<<<1, row>>>(res, arr,row, maxIndex );
	}
	else{
		cast_to_int<<<(int)(row/500)+1,500>>>(res, arr,row, maxIndex);
	}
	return res;
}
_global_ void copy_vec( float *a, float *b, int astart, int bstart, int dim ){

	int id = threadIdx.x+blockIdx.x*blockDim.x;
	if (id> dim){return ;}

	a[astart+id] = b[bstart+id];
};
float* accept_better(float*dobj, float*dsobj,
				   float* bests, float* newBests,
				   int nbgood, int dim, size_t pitch, curandGenerator_t gen){

	float obj  = (float) malloc( sizeof(float)*nbgood);
	float sobj = (float) malloc( sizeof(float)*nbgood);

	cudaMemcpy(obj, dobj, sizeof(float)*nbgood, cudaMemcpyDeviceToHost);
	cudaMemcpy(sobj, dsobj, sizeof(float)*nbgood, cudaMemcpyDeviceToHost);

	float *nobj, *merged;
	int nc=0;
	nobj =(float*) malloc( sizeof(float)*nbgood);
	merged = gen_random(gen, nbgood, dim, &pitch, 0.0,0.0);
	int oc=0, soc=0;

	while(nc<nbgood){
		if( obj[oc]>sobj[soc] ){
//			cout<<"------soc:"<<soc<<"--"<<soc*pitch<<"--"<<nc*pitch<<endl;
			nobj[nc] = sobj[soc];
//			cudaMemcpy(&merged[nc*pitch], &newBests[pitch*soc] , sizeof(float)*dim ,cudaMemcpyHostToHost);
//
			if(dim>1024){
				copy_vec<<<int(dim/500)+1, 500>>>(merged, newBests, pitch*nc,pitch*soc, dim);
			}else{

			copy_vec<<<1, dim>>>(merged, newBests, pitch*nc,pitch*soc, dim);
			}
			cudaDeviceSynchronize();
			++soc;
		}
		else {
			nobj[nc] = obj[oc];
//			cout<<"------oc:"<<oc<<"--"<<oc*pitch<<"--"<<nc*pitch<<endl;
//			cudaMemcpy(&merged[nc*pitch], &newBests[pitch*oc] , sizeof(float)*dim ,cudaMemcpyHostToHost);
			if(dim>1024){
				copy_vec<<<int(dim/500)+1, 500>>>(merged, bests, pitch*nc,pitch*oc, dim);
			}
			else{
			copy_vec<<<1, dim>>>(merged, bests, pitch*nc,pitch*oc, dim);
			}
			cudaDeviceSynchronize();
			++oc;
		}
		++nc;
	}
	cudaDeviceSynchronize();
	free(obj);
	free(sobj);
	cudaFree(bests);
//	cout<<"------out"<<endl;

	return merged;
//	cudaMemcpy(dobj, nobj, sizeof(float)*nbgood, cudaMemcpyHostToHost);
////	cudaMemcpy2D( bests, dim*sizeof(float), merged, pitch*sizeof(float),
////									dim*sizeof(float), nbgood, cudaMemcpyHostToHost);
//	bests = merged;
//	cudaDeviceSynchronize();

}
float avg_loss(float*obj, int nbgood){

	double sum =0.0;

	for(int i=0; i< nbgood;i++ )
		sum+=obj[i];
	return sum/nbgood;

}