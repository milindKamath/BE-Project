#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
#include <cuda.h>
#include <cstdlib>
#include <stdlib.h>
#include <bitset>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/find.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include "helper_timer.h"
 
#define MAX_CONF 100 
#define MIN_CONF 0 

#define MAX_SUPP 100 
#define MIN_SUPP 0 

#define WIDTH 55
#define MAX 500


using namespace std;	
using namespace thrust;


__global__ void RearrangeDataPatterns(int *d_array, int *d_new ,int *len,int *item, int size, int count){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx < count){
		//printf("My ID:\t%d\n", idx);
		int k = 0;
		int j = 0;
		while(k < size){
			for(int item_index = idx*WIDTH; item_index < idx*WIDTH+len[idx]; item_index++){
				if(d_array[item_index] == item[k]){
					d_new[idx*WIDTH+j] = item[k];
					j++; 
				}
			}
			k++;
		}
			//if (idx == 0){
			//int i = 0;
			//for (int index_col =idx*WIDTH; index_col <(idx+1)*WIDTH; index_col++){
			//	printf("[%d][%d]  %d..",idx, i++, d_new[56]);
			//	if(idx == 0)
			//		printf("\n");
			//}
			//}
		len[idx] = j ;
					 
	}
}


struct FileProps{
		fstream fp;
		long num_of_transactions ;

		FileProps(){
			num_of_transactions = 0 ;
		}

		//void fileProp(); 
}f_prop;

struct CmdArgs_Type {
		double confidence ;
		double support ;
		char* filename ;
		int conf_val;
		int supp_val; 

		CmdArgs_Type()
		{
			confidence = 80 ;
			support = 80 ;
		}
		//void clac_val();
		
}c_args;

struct compare_support{
	double supp;
	compare_support(double _supp) : supp(_supp) {};
	__host__ __device__
	bool operator()(int& x) const {
		return x < supp;
	}
};

struct ParentRefNode{
	int parentID;
	bool* bitRelation;	
	struct ParentRefNode *ptr;
	int count ;
	
	ParentRefNode(){parentID =0;
	ptr =NULL;};
	
	ParentRefNode(int size){
		this-> bitRelation = new bool [size];
		memset(this->bitRelation, 0, sizeof(bool) *size);
		this-> ptr = NULL;
		this-> count = 0;
	}
	
	void setParentID(int item){
		this->parentID = item;
	}
	
	void setBit(int index){
		this->bitRelation[index] =1;
		this-count++;
	}
};

struct GraphNode{
	int itemID;
	struct ParentRefNode **ParentList;
		
	void init (int item, int size){
		itemID = item;
		this -> ParentList = new ParentRefNode*[size+1];
		for (int index =0; index < size+1; index++)
			ParentList[index] = NULL; 
	}
	
	void init1 (int size){}
};

struct PatternElements{
	int element;
	int frequency;
	struct PatternElements *next;

	PatternElements(){element = 0;}

	PatternElements(int ele){
		element = ele;
		frequency = 1;
		next = NULL;
	}
};


struct CPpattern{
	int itemID;
	struct PatternElements *elements;
	struct CPpattern *next;

	CPpattern(int item){
		itemID = item;
		elements = NULL;
		next = NULL;
	}
};


struct String{
	string str;
	String *next;

	String(){
		str = "DEFAULT";
		next = NULL;
	}
};

void generateCPpattern(CPpattern*, GraphNode**, int, int);
void printCP(CPpattern*);

bool processCmdArgs(int argc, char* argv[]){
	
	if (argc < 2) {
		cerr << "USAGE: " << argv[0] << " FilePath\n" ;
		return false;
	}
	c_args.filename = argv[1] ;
	
	CONF :
	if (argc > 2){
	 c_args.confidence = atof(argv[2]);
	}
	if( c_args.confidence < MIN_CONF || c_args.confidence >= MAX_CONF ){
		cout << "Please enter a value between 1-100" ;
		goto CONF ;	
	}
	SUPP :
	if (argc > 3){
		c_args.support = atof(argv[3]) ;
	}
	if( c_args.support < MIN_CONF || c_args.support >= MAX_CONF ){
		cout << "Please enter a value between 1-100" ;
		goto SUPP ;
	}
		
	return true;
}


void fileProp(){			
	f_prop.fp.open(c_args.filename, ios::out | ios::in);
	if(f_prop.fp == NULL ){
		cout << "File Error : Please check the file path your entered.\n";
	}

	string line;
	while (getline(f_prop.fp,line))
		if (!line.empty())
			++ f_prop.num_of_transactions;
}
		
void calc_val(){
	c_args.conf_val=f_prop.num_of_transactions*(c_args.confidence/100);
	c_args.supp_val=f_prop.num_of_transactions*(c_args.support/100);
}


void rearrangeDataPatterns(int *trans_arr, int* item, int size, int* trans_length)
{
	int *d_array, *d_new, *d_len;
	
	cudaMalloc((void **)&d_array, f_prop.num_of_transactions* WIDTH* sizeof(int));
	cudaMemcpy(d_array, trans_arr, f_prop.num_of_transactions* WIDTH* sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMalloc((void **)&d_new, f_prop.num_of_transactions* WIDTH* sizeof(int));
	cudaMemset(d_new, 0, f_prop.num_of_transactions* WIDTH* sizeof(int));
	
	cudaMalloc((void **)&d_len, f_prop.num_of_transactions*sizeof(int));
	cudaMemcpy(d_len, trans_length, f_prop.num_of_transactions*sizeof(int), cudaMemcpyHostToDevice);
	
	dim3 GridDim =dim3 (ceil((f_prop.num_of_transactions/1024.0)),1,1);
	dim3 BlockDim = dim3 (1024,1,1);	
	RearrangeDataPatterns<<<GridDim, BlockDim>>>(d_array, d_new, d_len, item, size, f_prop.num_of_transactions);
	cudaDeviceSynchronize();
	cudaMemcpy(trans_arr, d_new, f_prop.num_of_transactions* WIDTH* sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(trans_length, d_len, f_prop.num_of_transactions*sizeof(int), cudaMemcpyDeviceToHost);
	
	/*for(int i=0;i<f_prop.num_of_transactions*WIDTH;i++){
		cout<<trans_arr[i]<<"\t";	
	}*/	
}


float timeForModule1=0.0;
float timeForModule2=0.0;
float timeForModule3=0.0;
float totaltimetaken=0.0;

int main (int argc, char* argv[]){
 
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		fprintf(stderr, "error: no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}
    	int dev = 0;
    	cudaSetDevice(dev);

    	cudaDeviceProp dev_prop;
    	if (cudaGetDeviceProperties(&dev_prop, dev) == 0){
    		printf("******************************GPU DETECTED*****************************\n");
    		printf("GPU Device Number                                    : %d\n", deviceCount);
			printf("GPU Device Name                                      : %s\n", dev_prop.name);
			printf("GPU Device Compute Capability                        : %d.%d\n", dev_prop.major, dev_prop.minor);
			printf("GPU Device Clock Rate                                : %d kHz\n", dev_prop.clockRate);
			printf("\n");
     	}
     	
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
     
	cout << "\n!!!!!!~~~~~~ GENERATING GFHTable ~~~~~~!!!!!!\n";
	printf("\n");
	
    
	if(!processCmdArgs(argc, argv))
		return 1;
	fileProp();
	
	calc_val();
	cout << "Total number of transactions = \t" << f_prop.num_of_transactions << endl;
	cout << "Minimum Support =\t" << c_args.supp_val << "\t= "<< c_args.support << "%\n";
	int *trans_arr = new int [f_prop.num_of_transactions*WIDTH];
	int *trans_length = new int[f_prop.num_of_transactions];
	
	fstream fp(c_args.filename, ios::out | ios::in);
	if(fp == NULL ){
		cout << "File Error : Please check the file path your entered.\n";
	}	
	string line;

	int itemize;
	int index= 0 ;
	int count = 0 ;
	while (getline(fp,line)){
		if (!line.empty()){
				int indexy = 0 ;
				index = count * WIDTH;
				istringstream stream(line);				
				while(stream >> itemize ){

					if (itemize != 0){
						trans_arr[index] = itemize;
						index++;
						indexy++;
					}	
				}
			trans_length[count++] = indexy ;
			
		}
	}
	device_vector<int>trans_vect(f_prop.num_of_transactions*55);
	copy_n(trans_arr, f_prop.num_of_transactions*WIDTH, trans_vect.begin());

   	sort(trans_vect.begin(), trans_vect.end());
	
	device_vector<int>item_arr(MAX);
	device_vector<int>freq_arrinit(f_prop.num_of_transactions*WIDTH,1);
	
	reduce_by_key(trans_vect.begin(), trans_vect.end(), make_constant_iterator<long int>(1), item_arr.begin(), freq_arrinit.begin());
	freq_arrinit.resize(MAX);
	sort_by_key(freq_arrinit.begin()+1,freq_arrinit.end(),item_arr.begin()+1,thrust::greater<int>());
	
	device_vector<int>::iterator iter;
	iter=find_if(freq_arrinit.begin(),freq_arrinit.end(),compare_support(c_args.supp_val));
	item_arr.resize(thrust::distance(freq_arrinit.begin(), iter));
	freq_arrinit.resize(thrust::distance(freq_arrinit.begin(), iter));
	 
	
	device_vector<int>item(thrust::distance(freq_arrinit.begin(), iter)-1);
	device_vector<int>freq(thrust::distance(freq_arrinit.begin(), iter)-1);
	
	thrust::copy(item_arr.begin()+1, item_arr.end(), item.begin());
	thrust::copy(freq_arrinit.begin()+1, freq_arrinit.end(), freq.begin());
	
	item_arr.clear();
	item_arr.shrink_to_fit();
	freq_arrinit.clear();
	freq_arrinit.shrink_to_fit();
	trans_vect.clear();
	trans_vect.shrink_to_fit();
	
	int *raw_ptr = raw_pointer_cast(item.data());
	int *raw_ptr1 = raw_pointer_cast(trans_vect.data());
	for(int index = 0; index<item.size(); index++)
		cout << "Item[" << index << "]" << item[index] << "\t->" << freq[index]<< endl;
		
	cout << "\n!!!!!!~~~~~~ GENERATED GFHTable ~~~~~~!!!!!!\n";
	
	
		
	rearrangeDataPatterns(trans_arr, raw_ptr, item.size(), trans_length);
	sdkStopTimer(&timer);
    timeForModule1 = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    
    cout<<"Time taken for GFHTable creation: "<<timeForModule1<<"ms\t==>"<<timeForModule1/1000<<" seconds\n";
    totaltimetaken=timeForModule1;
	
	host_vector<int>item1(item);
	
	
	//end of module 1
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------
	
	
	StopWatchInterface *timer1 = NULL;
    sdkCreateTimer(&timer1);
    sdkStartTimer(&timer1);	
		
	cout << "\n!!!!!!~~~~~~ START OF GRAPH CREATION ~~~~~~!!!!!!\n";	
	//for (int indext = 0; indext < f_prop.num_of_transactions; indext++)	  			
	//	cout << trans_length[indext]<< endl;
    
	GraphNode **graph = new GraphNode *[item1.size()];
	
	for(int index=0; index <item1.size(); index++){
		GraphNode *temp = new GraphNode();
		temp->init(item1[index], index);
		graph[index] = new GraphNode();
		graph[index] = temp;
	}
	int i;
	//for(int index=0; index <item1.size(); index++)
	//	cout << graph[index]-> itemID<< endl;
	ParentRefNode *temp;
	for (int indext = 0; indext < f_prop.num_of_transactions; indext++){
		GraphNode *parent = NULL;
		for (int index = 0; index < trans_length[indext]; index++){
			for (i =0; i < item1.size(); i++){
				if (item1[i] == trans_arr[indext*WIDTH+index]){
					if (parent == NULL){
						break;
					}
					else{
						if(graph[i]->ParentList[0] == NULL){
							temp = new ParentRefNode(f_prop.num_of_transactions);
							temp->setParentID(parent->itemID);
							temp->setBit(indext);
							graph[i]->ParentList[0] = temp;
//							cout << "Parent:\t" << graph[i]->ParentList[0]->parentID<<endl;
							break;		
						}
						
						else{
							for (int indexy =0; indexy < i; indexy++){ 
								if(graph[i]->ParentList[indexy] != NULL){
									if (graph[i]->ParentList[indexy]->parentID == parent->itemID){
										graph[i]->ParentList[indexy]->setBit(indext);
							//			cout << "COUNT:\t"<<graph[i]->ParentList[indexy]->count ;
										break;
									}
								}
								
								
								if(graph[i]->ParentList[indexy] == NULL){
									temp = new ParentRefNode(f_prop.num_of_transactions);
									temp->setParentID(parent->itemID);
									temp->setBit(indext);
									graph[i]->ParentList[indexy] = temp;
								//	cout << "Parent:\t" << graph[i]->ParentList[indexy]->parentID<<endl;
									break;
								}
								
							}
							break;
						}
							
					}
				}
			}
			parent = graph[i];
		}
	}
	
	cout<<endl<<item.size() -1 <<" nodes created\n";
/*	
	for(int index=0; index <item1.size(); index++){
	int ind = 0;
		while(graph[index]->ParentList[ind] != NULL){
			cout << "\n\n....." << graph[index]->itemID;
			cout << "\t"<< graph[index]->ParentList[ind]->parentID;
			cout << "\t"<< graph[index]->ParentList[ind]->count;
			ind++;
		}
	}
*/
		cout << "\n!!!!!!~~~~~~ GRAPH CREATED ~~~~~~!!!!!!\n";
		
		sdkStopTimer(&timer1);
    	timeForModule2 = sdkGetTimerValue(&timer1);
    	sdkDeleteTimer(&timer1);
    
   	    cout<<"Time taken for Graph creation: "<<timeForModule2<<"ms\t==>"<<timeForModule2/1000<<" seconds\n";
    	totaltimetaken=totaltimetaken+timeForModule2;
    
    
    	//end of module 2
    	//----------------------------------------------------------------------------------------------------------------------------------------------------------
    
	
		
		StopWatchInterface *timer2 = NULL;
    	sdkCreateTimer(&timer2);
    	sdkStartTimer(&timer2);
	
		CPpattern *head = NULL;
		CPpattern *curr= new CPpattern(0);
		CPpattern *nextnode;
		for (int i = 0; i < item1.size(); i++){
			if(head == NULL){
				head = new CPpattern(item1[i]);
				curr = head;
			}
			else{
				nextnode = new CPpattern(item1[i]);
				curr->next = nextnode;
				curr = nextnode;
			}
			generateCPpattern(curr,graph,i,item1.size());
		}	
		cout << "\n\n!!!!!!~~~~~~ CP PATTERNS GENERATED ~~~~~~!!!!!!\n";
		CPpattern *iterator = head;
		while(iterator != NULL){
			cout << iterator-> itemID;
			printCP(iterator);
			iterator = iterator-> next;
			cout << endl;
		}
		
	sdkStopTimer(&timer2);
    timeForModule3 = sdkGetTimerValue(&timer2);
    sdkDeleteTimer(&timer2);
    
   	cout<<"Time taken for CP Pattern creation: "<<timeForModule3<<"ms\t==>"<<timeForModule3/1000<<" seconds\n";
    totaltimetaken=totaltimetaken+timeForModule3;
    		
    cout<<"#################################################\n";
	cout<<"Time taken for Program execution: "<<totaltimetaken<<"ms\t==>"<<totaltimetaken/1000<<" seconds\n";
	}


void fillcp(CPpattern *curr, GraphNode **graph, int transid, int newindex, int size){
	PatternElements *newelement = new PatternElements(graph[newindex]->itemID);
	
	PatternElements *iter = curr->elements;
	if(curr->elements == NULL){
		curr-> elements = newelement;
		goto added;
	}
	while(iter-> next != NULL){
		iter = iter-> next;
	}
	iter-> next = newelement;

	// element added

	added: 
	
	if (graph[newindex]-> ParentList[0] == NULL)
		return;
	int nindex = 0;
	for (int index = 0; graph[newindex]-> ParentList[index] != NULL; index++){
		if(graph[newindex]-> ParentList[index]-> bitRelation[transid]){
			for (int j = 0; j < size; j++){
				if(graph[j]-> itemID==graph[newindex]-> ParentList[index]-> parentID){
					nindex = j;
					break;
				}
			}
			fillcp(curr, graph, transid, nindex, size);
		}
	}
	return;

}

void checkcp(CPpattern *curr, GraphNode **graph, int transid, int newindex, int size){
//	cout<<"Supestcheck"<<transid<<endl;
	PatternElements *iter = curr->elements;
	if(curr->elements == NULL){
		goto next;
	}	
	while(iter != NULL){
		if(iter-> element == graph[newindex]-> itemID){
			iter-> frequency++;
			goto next;
		}
		iter = iter-> next;
	}
	next:
//	int parentpath =0;
	if (graph[newindex]-> ParentList[0] == NULL)
		return;
	int nindex = 0;
	for (int index = 0; graph[newindex]-> ParentList[index] != NULL; index++){
		if(graph[newindex]-> ParentList[index]-> bitRelation[transid]){
			for (int id = 0; id < size; id++){
				if(graph[id]-> itemID == graph[newindex]-> ParentList[index]->  parentID){
					nindex = id;
					break;
				}
			}
			checkcp(curr,graph,transid,nindex, size);
		}
	}
}


void pruneCP(CPpattern* curr, int numPaths){
	PatternElements *current, *previous;

	current = curr-> elements;
	previous = NULL;

	while(current != NULL){
		if (current-> frequency != numPaths){
			if(previous == NULL){
				curr-> elements = current-> next;
				current-> next = NULL;
				current = curr-> elements;
			}
			else{
				previous-> next = current-> next;
				current = current-> next;
			}
		}
		else{
			previous = current;
			current = current-> next;
		}
	}
}



void generateCPpattern(CPpattern *curr, GraphNode **graph, int i, int size){
	if(graph[i]-> ParentList[0]==NULL){
		return;
	}

	int multipath = 0;
	int newindex = 0;
	bool firsttime = true;
/*
	for (int index = 0; graph[i]->ParentList[index]!=NULL; index++){
		cout << graph[i]-> itemID;
	}
	cout << endl<<endl;
*/
	for (int index = 0; graph[i]-> ParentList[index] != NULL; index++){
		for (int j = 0; j < size; j++){
			if(graph[j]-> itemID == graph[i]-> ParentList[index]-> parentID){
				newindex = j;
				break;
			}
		}
		int pathcounts = graph[i]-> ParentList[index]-> count;
		multipath += pathcounts;
		int transid = 0;
		int count = 0;
		while(pathcounts != 0){
			while(count < f_prop.num_of_transactions){
				if(graph[i]-> ParentList[index]-> bitRelation[count]){
					transid = count;
					break;
				}
				count++;
			}
			if(firsttime){
				fillcp(curr, graph, transid, newindex, size);
				firsttime = false;
			}
			else{
				checkcp(curr, graph, transid, newindex, size);
			}	
			count++;
			pathcounts--;
		}
	}

	if(multipath > 1){
		pruneCP(curr, multipath);
	}
}

int generateCombinations(string *inputSet, int inIndex, string prefixSet, string** outputSet, int outIndex, int count){
	string tempSet = "";
	int localIndex = inIndex;
	int index = 0;
	while(localIndex < count){
		tempSet = prefixSet + inputSet[localIndex] + " " ;
		outputSet[outIndex][index] = tempSet;
		outIndex = generateCombinations(inputSet, localIndex+1, tempSet, outputSet, outIndex+1, count); 
		localIndex++;
	}
	return outIndex;
} 

void getCombinations(string inputSet, int count, CPpattern *iter){
	if(inputSet.length() == 0)
		return;
	string* inputSetSend = new string [count];
	int inIndex = 0;
	for ( int index = 0; index < count ;){
		if(inputSet[inIndex] != ' ')	
			inputSetSend[index] += inputSet[inIndex];
		else 
			index++; 
		inIndex++;
	}
			
	string **outputSet;
	int len =  pow(2, count) - 1;
	outputSet = new string* [len];
	for ( int index = 0; index < len ; index++ )
		outputSet[index] = new string;
	generateCombinations (inputSetSend, 0, "", outputSet, 0, count);
	for ( int index = 0; index < len ; index++ ){
		cout << "{" << iter-> itemID << " " <<  outputSet[index][0]  << "}"<< endl;
	}
}

void printCP(CPpattern *iter){
	cout << "-> ";
	cout << "{" << iter-> itemID << "}";
	string patternSet;
	int count = 0;
	PatternElements *indexIter = iter-> elements;
	while(indexIter != NULL){
		stringstream ss;
		ss << indexIter-> element;
		patternSet += ss.str();
		patternSet += " ";
		count ++;
		indexIter = indexIter-> next;
	}
//	cout << patternSet <<endl ;
	
	getCombinations(patternSet, count, iter);
}
