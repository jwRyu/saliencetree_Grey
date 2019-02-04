#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include<math.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/times.h>
#include <unistd.h>
#include <assert.h>


#define BOTTOM (-1)
#define false 0
#define true  1
#define MIN(a,b)  ((a<=b) ? (a) : (b))
#define MAX(a,b)  ((a>=b) ? (a) : (b))

#define CONNECTIVITY  4

typedef short bool;
typedef unsigned char ubyte;


int width, height, size;  

int arealambda, colourlambda=0;
double omegafactor = 200.0;
double LABweight[3] = {1.00, 1.00, 1.0};
double RGBweight[3] = {1.00, 0.2, 0.0};

typedef ubyte Pixel[3];
typedef float LABpixel[3];

Pixel *gval=NULL, *out=NULL;
LABpixel *labimg;

typedef struct Edge{
  int p, q;
  double alpha;
} Edge;

typedef struct{
  int size,maxsize;
  Edge *queue;
} EdgeQueue;

EdgeQueue *EdgeQueueCreate(long maxsize){
  EdgeQueue *newQueue = (EdgeQueue *) malloc(sizeof(EdgeQueue));
  newQueue->size = 0;
  newQueue->queue = (Edge *)malloc((maxsize+1)*sizeof(Edge));
  newQueue->maxsize=maxsize;
  return newQueue;
}



#define EdgeQueueFront(queue)       (queue->queue + 1)
#define IsEmpty(queue)        ((queue->size)==0)

void EdgeQueueDelete(EdgeQueue *oldqueue){
  free(oldqueue->queue);
  free(oldqueue);

}



void EdgeQueuePop(EdgeQueue *queue ){
  int current = 1;
  Edge moved;

  moved.p = queue->queue[queue->size].p;
  moved.q = queue->queue[queue->size].q;
  moved.alpha = queue->queue[queue->size].alpha;
  
  queue->size--;


  while ( ((current*2<=queue->size) &&
	   (moved.alpha > queue->queue[current*2].alpha))
	  ||
	  ((current*2+1<=queue->size) &&
	   (moved.alpha > queue->queue[current*2+1].alpha))
	  ){
    if ((current*2+1<=queue->size) && 
	(queue->queue[current*2].alpha > 
	 queue->queue[current*2+1].alpha)){
      queue->queue[current].p = queue->queue[current*2+1].p;
      queue->queue[current].q = queue->queue[current*2+1].q;
      queue->queue[current].alpha = queue->queue[current*2+1].alpha;
      current+=current+1;
    } else {
      queue->queue[current].p = queue->queue[current*2].p;
      queue->queue[current].q = queue->queue[current*2].q;
      queue->queue[current].alpha = queue->queue[current*2].alpha;
      current+=current;
    }
  }
  queue->queue[current].p=moved.p;
  queue->queue[current].q=moved.q;
  queue->queue[current].alpha=moved.alpha;

}

void EdgeQueuePush(EdgeQueue *queue, int p, int q, double alpha){
  long current;
 
  queue->size++;
  current=queue->size;
  
  while ((current/2 !=0) && (queue->queue[current/2].alpha > alpha)){
    queue->queue[current].p= queue->queue[current/2].p;
    queue->queue[current].q= queue->queue[current/2].q;
    queue->queue[current].alpha= queue->queue[current/2].alpha;
    current=current/2;
  }

  queue->queue[current].p=p;
  queue->queue[current].q=q;
  queue->queue[current].alpha=alpha;

}

typedef struct SalienceNode  
{ 
  int parent;
  int area;
  bool filtered; /* indicates whether or not the filtered value is OK */
  Pixel outval;  /* output value after filtering */
  double alpha;  /* alpha of flat zone */
  double sumPix[3];
  Pixel minPix;
  Pixel maxPix;
} SalienceNode;



typedef struct SalienceTree {
  int maxSize;
  int curSize;
  SalienceNode *node;
} SalienceTree;


SalienceTree *CreateSalienceTree(int imgsize){
  SalienceTree *tree = malloc(sizeof(SalienceTree));
  tree->maxSize = 2*imgsize;  /* potentially twice the number of nodes as pixels exist*/
  tree->curSize = imgsize;    /* first imgsize taken up by pixels */
  tree->node = malloc(2*imgsize*sizeof(SalienceNode));
  return tree;
}

void DeleteTree(SalienceTree *tree){
  free(tree->node);
  free(tree);
}

int NewSalienceNode(SalienceTree *tree, int *root, double alpha){
  SalienceNode *node =tree->node + tree->curSize;
  int result = tree->curSize;
  tree->curSize++;
  node->alpha=alpha;
  node->parent=BOTTOM;
  root[result]=BOTTOM;
  return result;
}

void MakeSet(SalienceTree *tree, int *root, Pixel *gval, int p){
  int i;
  tree->node[p].parent=BOTTOM;
  root[p]=BOTTOM;
  tree->node[p].alpha = 0.0;
  tree->node[p].area = 1;
  for (i=0;i<3;i++){
    tree->node[p].sumPix[i] = gval[p][i];
    tree->node[p].minPix[i] = gval[p][i];
    tree->node[p].maxPix[i] = gval[p][i];
  }
}

int FindRoot(int *root, int p){
  int r=p, i,j;

  while (root[r]!=BOTTOM){
    r=root[r];
  }
  i=p;
  while (i!=r){
    j=root[i];
    root[i]=r;
    i=j;
  }
  return r;
}

int FindRoot1(SalienceTree *tree, int *root, int p){
  int r=p, i,j;

  while (root[r]!=BOTTOM){
    r=root[r];
  }
  i=p;
  while (i!=r){
    j=root[i];
    root[i]=r;
    tree->node[i].parent = r;
    i=j;
  }
  return r;
}

double simpleSalience(Pixel p, Pixel q){
  double result=0; 
  int i;

  for (i=0;i<3;i++)
    result+= ((double)p[i] -(double)q[i])*((double)p[i] -(double)q[i]);
  return sqrt(result);
}

double WeightedSalience(Pixel p, Pixel q){
  double result=0; 
  int i;

  for (i=0;i<3;i++)
    result+= RGBweight[i]*((double)p[i] -(double)q[i])*((double)p[i] -(double)q[i]);
  return sqrt(result);
}

void RGBtoLAB(Pixel p, LABpixel q){
  double r,g,b,X,Y,Z,Xw,Yw,Zw;

   
    r  = p[0];
    g  = p[1];
    b  = p[2];



	if ( r > 0.04045 ) r = pow( ( ( r + 0.055 ) / 1.055 ), 2.4);
		else r = r / 12.92;

	if ( g > 0.04045 ) g = pow( ( ( g + 0.055 ) / 1.055 ), 2.4);
		else g = g / 12.92;

	if ( b > 0.04045 ) b = pow( ( ( b + 0.055 ) / 1.055 ), 2.4);
		else b = b / 12.92;


	X = r * 0.4124 + g * 0.3576 + b * 0.1805;
	Y = r * 0.2127 + g * 0.7152 + b * 0.0722;
	Z = r * 0.0193 + g * 0.1192 + b * 0.9502;

// CONVERSION FROM XYZ TO LAB STARTS

	//reference point is color white.

	Xw = 0.95045;
	Yw = 1.0000;
	Zw = 1.08892;

	X = X / Xw;
	Y = Y / Yw;
	Z = Z / Zw;

	if ( X > 0.008856 ) X = pow(X, 1.0/3.0 );
		else     X = ( 7.787 * X ) + ( 16.0 / 116.0 );
	if ( Y > 0.008856 ) Y = pow(Y, 1.0/3.0 );
		else     Y = ( 7.787 * Y ) + ( 16.0 / 116.0 );
	if ( Z > 0.008856 ) Z = pow(Z, 1.0/3.0 );
		else     Z = ( 7.787 * Z ) + ( 16.0 / 116.0 );

	q[0] = ( 116.0 * Y ) - 16.0;
	q[1] = 500.0 * ( X - Y );
	q[2] = 200.0 * ( Y - Z );


} /* RGBtoLAB */

void MakeLABImage (Pixel *gval, LABpixel *labimage, int size){
  int i;
  for (i=0;i<size;i++)
    RGBtoLAB(gval[i],labimage[i]);
}

double LABSalience(LABpixel p, LABpixel q ){
  double result=0; 
  int i;

  for (i=0;i<3;i++)
    result+= ((double)p[i] -(double)q[i])*((double)p[i] -(double)q[i]);
  return sqrt(result);
}
double WeightedLABSalience(LABpixel p, LABpixel q){
  double result=0; 
  int i;

  for (i=0;i<3;i++)
    result+= LABweight[i]*((double)p[i] -(double)q[i])*((double)p[i] -(double)q[i]);
  return sqrt(result);
}

bool IsLevelRoot(SalienceTree *tree, int i){
  int parent = tree->node[i].parent;

  if (parent==BOTTOM)
    return true;
  return (tree->node[i].alpha != tree->node[parent].alpha);
}

int LevelRoot(SalienceTree *tree, int p){
  int r=p, i,j;

  while (!IsLevelRoot(tree,r)){
    r=tree->node[r].parent;
  }
  i=p;

  while (i!=r){
    j=tree->node[i].parent;
   tree->node[i].parent=r;
    i=j;
  }
  return r;
}

# define Par(tree,p) LevelRoot(tree,tree->node[p].parent)
 

void Union(SalienceTree *tree,int *root, int p, int q){ /* p is always current pixel */
  int i;
  
  q=FindRoot1(tree,root,q);
  
  if (q!=p){
    tree->node[q].parent = p;
    root[q] = p;
    tree->node[p].area +=  tree->node[q].area;
    for (i=0;i<3;i++){
      tree->node[p].sumPix[i] += tree->node[q].sumPix[i];
      tree->node[p].minPix[i] = MIN(tree->node[p].minPix[i],tree->node[q].minPix[i]);
      tree->node[p].maxPix[i] = MAX(tree->node[p].maxPix[i],tree->node[q].maxPix[i]);
    }
  }
}


void Union2(SalienceTree *tree, int *root, int p, int q){ 
  int i;
  tree->node[q].parent = p;
  root[q] = p;
  tree->node[p].area +=  tree->node[q].area;
  for (i=0;i<3;i++){
    tree->node[p].sumPix[i] += tree->node[q].sumPix[i];
    tree->node[p].minPix[i] = MIN(tree->node[p].minPix[i],tree->node[q].minPix[i]);
    tree->node[p].maxPix[i] = MAX(tree->node[p].maxPix[i],tree->node[q].maxPix[i]);
  }
}

void Phase1(SalienceTree *tree, EdgeQueue *queue, int *root,
	    Pixel *img, LABpixel *labimg, 
	    int width, int height, double alphamin,
	    double (*salience)(LABpixel, LABpixel)){
  /* pre: tree has been created with imgsize= width*height 
          queue initialized accordingly;
   */
  int imgsize = width*height;
  int p,x,y;

 
  MakeSet(tree, root, img, 0);
  
  for (x=1;x<width;x++){
    MakeSet(tree, root, img, x);
    if ((*salience)(labimg[x],labimg[x-1])<alphamin){
      Union(tree, root, x, x-1);
    } else {
      EdgeQueuePush(queue,x,x-1,(*salience)(labimg[x],labimg[x-1]));
    }
  }

  for (y = 1; y<height; y++){
    p=y*width;
    MakeSet(tree, root, img, p);
    if ((*salience)(labimg[p],labimg[p-width])<alphamin){
      Union(tree, root, p, p-width);
    } else {
      EdgeQueuePush(queue,p, p-width,(*salience)(labimg[p],labimg[p-width]));
    }
    p++;  
    for (x=1;x<width;x++,p++){
      MakeSet(tree, root, img, p);
      if ((*salience)(labimg[p],labimg[p-width])<alphamin){
	Union(tree, root, p, p-width);
      } else {
	EdgeQueuePush(queue,p, p-width,(*salience)(labimg[p],labimg[p-width]));
      }
      if ((*salience)(labimg[p],labimg[p-1])<alphamin){
	Union(tree, root, p, p-1);
      } else {
	EdgeQueuePush(queue,p, p-1,(*salience)(labimg[p],labimg[p-1]));
      }
      
    }    

  }
  
}


void GetAncestors(SalienceTree *tree, int *root, int *p, int*q){
  int temp;
  *p = LevelRoot(tree, *p);
   *q = LevelRoot(tree, *q);
  if (*p<*q){
    temp = *p;
    *p=*q;
    *q=temp;
  }
  while ((*p!=*q) && (root[*p]!=BOTTOM) &&(root[*q]!=BOTTOM)){
    *q=root[*q];
    if (*p<*q){
      temp = *p;
      *p=*q;
      *q=temp;
    }
  }
  if (root[*p]==BOTTOM) {
    *q=FindRoot(root,*q);
  } else if (root[*q]==BOTTOM){
    *p=FindRoot(root,*p);
  }
}

void Phase2(SalienceTree *tree, EdgeQueue *queue, int *root,
	    Pixel *img, 
	    int width, int height){
  Edge *currentEdge;
  int v1, v2, temp, r;
  double oldalpha=0,alpha12; 
  while (!IsEmpty(queue)){
    currentEdge = EdgeQueueFront(queue);
    v1 =currentEdge->p;
    v2 =currentEdge->q;
    GetAncestors(tree,root,&v1,&v2);
    alpha12 = currentEdge->alpha;

    EdgeQueuePop(queue);
    if (v1!=v2) {
      if (v1 < v2) {
	temp=v1;
	v1=v2;
	v2=temp;
      }
      if (tree->node[v1].alpha < alpha12){
	r =NewSalienceNode(tree,root, alpha12);
	Union2(tree,root,r,v1);
	Union2(tree,root,r,v2);
      } else {
	Union2(tree,root,v1,v2);
      }
    }
    oldalpha=alpha12;
  }
}

SalienceTree *MakeSalienceTree(Pixel *img, LABpixel *labimg, 
			       int width, int height, double alphamin){
  int imgsize = width*height;
  EdgeQueue *queue = EdgeQueueCreate((CONNECTIVITY/2)*imgsize);
  int *root = malloc(imgsize*2*sizeof(int));
  SalienceTree *tree;
  tree = CreateSalienceTree(imgsize);
  assert(tree!=NULL);
  assert(tree->node!=NULL);
  fprintf(stderr,"Phase1 started\n");
  Phase1(tree,queue,root,img,labimg,width,height,alphamin,WeightedLABSalience);
  fprintf(stderr,"Phase2 started\n");
  Phase2(tree,queue,root,img,width,height);
  fprintf(stderr,"Phase2 done\n");
  EdgeQueueDelete(queue);
  free(root);
  return tree;
}


void SalienceTreeAreaFilter(SalienceTree *tree, Pixel *out, int lambda){
  int i,j, imgsize = tree->maxSize/2;
  if (lambda <= imgsize){
    for (j=0;j<3;j++){
      tree->node[tree->curSize-1].outval[j] =
      tree->node[tree->curSize-1].sumPix[j]/tree->node[tree->curSize-1].area;
    }
    for (i = tree->curSize-2; i>=0; i--){
      
      if (IsLevelRoot(tree,i) && (tree->node[i].area>=lambda)){
	for (j=0;j<3;j++)
	  tree->node[i].outval[j] = tree->node[i].sumPix[j]/tree->node[i].area;
      } else {
	for (j=0;j<3;j++)
	  tree->node[i].outval[j] = tree->node[tree->node[i].parent].outval[j];
      }
    }
  } else {
    for (i = tree->curSize-1; i>=0; i--){
      for (j=0;j<3;j++)
	tree->node[i].outval[j] = 0;
    }
  }
  for (i = 0; i < imgsize; i++)
    for (j=0;j<3;j++)
      out[i][j] = tree->node[i].outval[j];
}

#define NodeSalience(tree, p) (tree->node[Par(tree,p)].alpha)

void SalienceTreeSalienceFilter(SalienceTree *tree, Pixel *out, double lambda){
  int i,j, imgsize = tree->maxSize/2;
  if (lambda <= tree->node[tree->curSize-1].alpha){
    for (j=0;j<3;j++){
      tree->node[tree->curSize-1].outval[j] =
      tree->node[tree->curSize-1].sumPix[j]/tree->node[tree->curSize-1].area;
    }
    for (i = tree->curSize-2; i>=0; i--){
      
      if (IsLevelRoot(tree,i) && (NodeSalience(tree,i)>=lambda)){
	for (j=0;j<3;j++)
	  tree->node[i].outval[j] = tree->node[i].sumPix[j]/tree->node[i].area;
      } else {
	for (j=0;j<3;j++)
	  tree->node[i].outval[j] = tree->node[tree->node[i].parent].outval[j];
      }
    }
  } else {
    for (i = tree->curSize-1; i>=0; i--){
      for (j=0;j<3;j++)
	tree->node[i].outval[j] = 0;
    }
  }
  for (i = 0; i < imgsize; i++)
    for (j=0;j<3;j++)
      out[i][j] = tree->node[i].outval[j];

}


short ImagePPMAsciiRead(char *fname)
{
   FILE *infile;
   ulong i,j;
   int c;

   infile = fopen(fname, "r");
   if (infile==NULL) {
      fprintf (stderr, "Error: Can't read the ASCII file: %s !", fname);
      return(0);
   }
   fscanf(infile, "P3\n");
   while ((c=fgetc(infile)) == '#')
      while ((c=fgetc(infile)) != '\n');
   ungetc(c, infile);
   fscanf(infile, "%d %d\n255\n", &width, &height);
   size = width*height;

   gval = malloc(size*sizeof(Pixel));
   if (gval==NULL) {
      fprintf (stderr, "Out of memory!");
      fclose(infile);
      return(0);
   }
   for (i=0; i<size; i++)
   {
     for (j=0;j<3;j++){
       fscanf(infile, "%d", &c);
       gval[i][j] = c;
     }
   }
   fclose(infile);
   return(1);
} /* ImagePGMAsciiRead */


short ImagePPMBinRead(char *fname)
{
   FILE *infile;
   int c, i;

   infile = fopen(fname, "rb");
   if (infile==NULL) {
      fprintf (stderr, "Error: Can't read the binary file: %s !", fname);
      return(0);
   }
   fscanf(infile, "P6\n");
   while ((c=fgetc(infile)) == '#')
      while ((c=fgetc(infile)) != '\n');
   ungetc(c, infile);
   fscanf(infile, "%d %d\n255\n", &width, &height);
   size = width*height;

   gval = malloc(size*sizeof(Pixel));
   if (gval==NULL) {
     fprintf (stderr, "Out of memory!");
     fclose(infile);
     return(0);
   } 
   fread(gval, sizeof(Pixel), size, infile);

   fclose(infile);
   return(1);
} /* ImagePGMBinRead */


short ImagePPMRead(char *fname)
{
   FILE *infile;
   char id[4];

   infile = fopen(fname, "r");
   if (infile==NULL) {
      fprintf (stderr, "Error: Can't read the image: %s !", fname);
      return(0);
   }
   fscanf(infile, "%3s", id);
   fclose(infile);
   if (strcmp(id, "P3")==0) return(ImagePPMAsciiRead(fname));
   else if (strcmp(id, "P6")==0) return(ImagePPMBinRead(fname));
   else {
     fprintf (stderr, "Unknown type of the image!");
     return(0);
   }
} /* ImagePPMRead */

int ImagePPMBinWrite(char *fname)
{
   FILE *outfile;

   outfile = fopen(fname, "wb");
   if (outfile==NULL) {
      fprintf (stderr, "Error: Can't write the image: %s !", fname);
      return(-1);
   }
   fprintf(outfile, "P6\n%d %d\n255\n", width, height);

   fwrite(out,sizeof(Pixel) , (size_t)(size), outfile);

   fclose(outfile);
   return(0);
} /* ImagePPMBinWrite */


int main (int argc, char *argv[]) {
  
   char *imgfname, *outfname = "out.ppm";
   int r;
   ulong i;
   clock_t start;
   struct tms tstruct;
   long tickspersec = sysconf(_SC_CLK_TCK);  
   float musec;
   SalienceTree *tree;

   if (argc<3)
   {
      printf("Usage: %s <input image> <area threshold> [salience threshold]  [omegafactor] [output image] \n", argv[0]);
      exit(0);
   }

   imgfname = argv[1];

   arealambda = atoi(argv[2]);
   if (argc>3)  colourlambda = atof(argv[3]);

   if (argc>4)  omegafactor = atof(argv[4]);

   if (argc>5)  outfname = argv[5];
 
   
   if (!ImagePPMRead(imgfname)) 
     return(-1);

   out = malloc(size*sizeof(Pixel));
   labimg = malloc(size*sizeof(LABpixel));

   printf("Filtering image '%s' using attribute area with lambda=%d, and edge salience threshold = %d\n", imgfname, arealambda, colourlambda);
   printf("Image: Width=%d Height=%d\n", width, height);
   
   printf("Data read, start filtering.\n");
   start = times(&tstruct);
   MakeLABImage(gval,labimg,size);
   musec = (float)(times(&tstruct) - start)/((float)tickspersec);
   printf("LAB conversion: wall-clock time: %f s\n",musec);
  
   tree = MakeSalienceTree(gval,labimg,width,height,(double)colourlambda);

   musec = (float)(times(&tstruct) - start)/((float)tickspersec);

   printf("Tree built: wall-clock time: %f s\n",musec);
   SalienceTreeAreaFilter(tree,out,arealambda);
     /*SalienceTreeSalienceFilter(tree,out,(double)lambda);*/
   
   musec = (float)(times(&tstruct) - start)/((float)tickspersec);

   printf("Tree Filtered: wall-clock time: %f s\n",musec);
   
   r = ImagePPMBinWrite(outfname);
   free(out);  
   if (r)  printf("Filtered image written to '%s'\n", outfname);
   
   free(gval);
   return(0);
} /* main */
