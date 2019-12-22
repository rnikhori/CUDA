
#include<bits/stdc++.h>

using namespace std;
const float pi=3.14;
int N_iter;

//simplebounds function

double fun(double u[])
{
	double z=(u[0]*u[0])+(u[1]*u[1])+(u[2]*u[2]);
	return z;
}

__global__ void simplebounds(double *s, int  lb[], int ub[],int d){
	for(int i=0; i<d; ++i){

		if(s[i]<lb[i]) s[i] = lb[i];
		if(s[i]>ub[i]) s[i] = ub[i];
	}
}

//levy and epsilon
__global__ void flower_pollination()
{
	if (rand()<p)       
			{
				 //no call to levi do it here

				int i,d=3;
				float beta=1.5;
				double u[d];
				double v[d];
				//double step[d],l[d];
				double *step = new double[d];
				double sigma=pow((tgamma(1+beta)*sin(pi*beta/2)/(tgamma((1+beta)/2)*beta*pow(2,((beta-1)/2)))),(1/beta));

				for(i=0;i<d;i++)
				{
					u[i]=(curand())*sigma;
				}

				for(i=0;i<d;i++)
				{
					v[i]=curand();
				}

				for(i=0;i<d;i++)
				{
				step[i]=u[i]/pow(abs(v[i]),(1/beta));
				step[i]=0.01*step[i];
				}
				
				for(j=0;j<d;j++)
				{

		           double dS=(step[j])*(sol[i][j]-best[j]);
				   sol[i][j]=sol[i][j]+dS;
				}
			}

			else
			{
				int epsilon=curand();

				int  jk[n];
				for(int z=0;z<n;z++)
					jk[z]=z;
				for(int z=0;z<n;z++)
				{
					int a,b;
					a=curand()%(n-z)+z;
					b=jk[a];
					jk[a]=jk[z];
					jk[z]=b;
				}

				for(j=0;j<d;j++)
				{

					s[i][j]=s[i][j]+epsilon*(sol[jk[1]][j]-sol[jk[2]][j]);
				}
			}
} 
//optimal solution

__global__ optimal_sol()
{
	for(j=0;j<d;j++)
		{
			arr[j]=s[i][j];
		}

	fnew=(u[0]*u[0])+(u[1]*u[1])+(u[2]*u[2]);

			if(fnew<=fitness[i]) //kernel
			{
				for (j=0;j<d;j++)
					sol[i][j]=s[i][j];
				fitness[i]=fnew;
			}
			if(fnew<min)
			{
				for (j=0;j<d;j++)
					best[j]=s[i][j];
				min=fnew;
			}

			for(j=0;j<d;j++)
			{
				cout<<sol[i][j]<<" ";

			}
			cout<<endl;
}

int main()
{
	int d=3,i,j,n,t;

	cout<<"enter value of N and  Iterations"<<endl;
	cin>>n>>N_iter;

	float p=0.8;
	double s[n][d],sol[n][d];
	int lb[d], ub[d];

	for(j=0;j<3;j++)
	{
		lb[j]=-100;
	}
	for(j=0;j<3;j++)
	{
		ub[j]=100;
	}

	double fitness[n],arr[d];

	//initialization using fitness

	for (i=0;i<n;i++)
	{
		for(j=0;j<3;j++)
		{
			sol[i][j]=(double)((rand()%10)/10.0);
			sol[i][j]=lb[j]+((ub[j]-lb[j])*(sol[i][j]));

			arr[j]=sol[i][j];

		}
		fitness[i]=fun(arr);
	}


	//finding best and minimum value

	double min=fitness[0];
	int k=0;
	for(i=1;i<n;i++)
	{
		if(fitness[i]<min)
		{
			min=fitness[i];
			k=i;
		}
	}

	cout<<"fitness min index"<<k<<endl;
	double best[3];
	for(j=0;j<3;j++)
	{
		best[j]=sol[k][j];
		cout<<best[j]<<" ";
	}
	cout<<endl;

	for (i=0;i<n;i++)
	{
		for (j=0;j<d;j++)
		{
			s[i][j] = sol[i][j];
		}
	}

//upto here run in host

	//two kernels from temp.cu
	flower_pollination<<<n,1>>>();
	simplebounds<<<1,1>>>(s[i], lb, ub,d);
			//Evaluating  new solutions
			double fnew;
	optimal_sol<<<1,1>>>();


	cout<<"Total number of evaluations: "<<N_iter*n<<endl;
	cout<<" fmin="<<min<<endl;
	cout<<"best values"<<endl;
	for(i=0;i<d;i++)
		cout<<" "<<best[i]<<" ";
	cout<<endl;

}

