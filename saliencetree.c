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

int lambda;
double omegafactor = 2.0;

typedef ubyte Pixel;

Pixel *gval=NULL, *out=NULL;

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
  newQueue->bucket = (Edge *)malloc((maxsize+1)*sizeof(Edge));
  newQueue->maxsize=maxsize;
  return newQueue;
}



#define EdgeQueueFront(queue)       (queue->queue + 1)
#define IsEmpty(queue)        ((queue->size)==0)

void EdgeQueueDelete(EdgeQueue *oldqueue){
  free(oldqueue->bucket);
  free(oldqueue);

}



void EdgeQueuePop(EdgeQueue *queue ){
  int current = 1;
  Edge moved;

  moved.p = queue->bucket[queue->size].p;
  moved.q = queue->bucket[queue->size].q;
  moved.alpha = queue->bucket[queue->size].alpha;
  
  queue->size--;


  while ( ((current*2<=queue->size) &&
	   (moved.alpha > queue->bucket[current*2].alpha))
	  ||
	  ((current*2+1<=queue->size) &&
	   (moved.alpha > queue->bucket[current*2+1].alpha))
	  ){
    if ((current*2+1<=queue->size) && 
	(queue->bucket[current*2].alpha > 
	 queue->bucket[current*2+1].alpha)){
      queue->bucket[current].p = queue->bucket[current*2+1].p;
      queue->bucket[current].q = queue->bucket[current*2+1].q;
      queue->bucket[current].alpha = queue->bucket[current*2+1].alpha;
      current+=current+1;
    } else {
      queue->bucket[current].p = queue->bucket[current*2].p;
      queue->bucket[current].q = queue->bucket[current*2].q;
      queue->bucket[current].alpha = queue->bucket[current*2].alpha;
      current+=current;
    }
  }
  queue->bucket[current].p=moved.p;
  queue->bucket[current].q=moved.q;
  queue->bucket[current].alpha=moved.alpha;

}

void EdgeQueuePush(EdgeQueue *queue, int p, int q, double alpha){
  long current;
 
  queue->size++;
  current=queue->size;
  
  while ((current/2 !=0) && (queue->bucket[current/2].alpha > alpha)){
    queue->bucket[current].p= queue->bucket[current/2].p;
    queue->bucket[current].q= queue->bucket[current/2].q;
    queue->bucket[current].alpha= queue->bucket[current/2].alpha;
    current=current/2;
  }

  queue->bucket[current].p=p;
  queue->bucket[current].q=q;
  queue->bucket[current].alpha=alpha;

}

typedef struct SalienceNode  
{ 
  int parent;
  int area;
  bool filtered; /* indicates whether or not the filtered value is OK */
  Pixel outval;  /* output value after filtering */
  double alpha;  /* alpha of flat zone */
  double sumPix;
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

int NewSalienceNode(SalienceTree *tree, double alpha){
  SalienceNode *node =tree->node + tree->curSize;
  int result = tree->curSize;
  tree->curSize++;
  node->alpha=alpha;
  node->parent=BOTTOM;
  return result;
}

void MakeSet(SalienceTree *tree, Pixel *gval, int p){

  tree->node[p].parent=BOTTOM;
  tree->node[p].alpha = 0.0;
  tree->node[p].area = 1;
  tree->node[p].sumPix = gval[p];
  tree->node[p].minPix = gval[p];
  tree->node[p].maxPix = gval[p];
}

int FindRoot(SalienceTree *tree, int p){
  int r=p, i,j;

  while (tree->node[r].parent!=BOTTOM){
    r=tree->node[r].parent;
    assert(r<tree->maxSize);
  }
  i=p;
  while (i!=r){
    j=tree->node[i].parent;
    tree->node[i].parent=r;
    i=j;
  }
  return r;
}

double simpleSalience(Pixel p, Pixel q){
  return fabs((double)p - (double)q);
}

bool IsLevelRoot(SalienceTree *tree, int i){
  int parent = tree->node[i].parent;

  if (parent==BOTTOM)
    return true;
  return ((tree->node[i].alpha != tree->node[parent].alpha)&&
	  tree->node[i].alpha*omegafactor >= simpleSalience(tree->node[i].minPix,tree->node[i].maxPix));
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
 
int FindRoot2(SalienceTree *tree, int p){
  int r=p;

  while (tree->node[r].parent!=BOTTOM){
    r=Par(tree,r);
     
  }
  return r;
}

void Union(SalienceTree *tree, int p, int q){ /* p is always current pixel */
     
  
  q=FindRoot(tree,q);
  
  if (q!=p){
    tree->node[q].parent = p;
    tree->node[p].area +=  tree->node[q].area;
    tree->node[p].sumPix += tree->node[q].sumPix;
  }
}


void Union2(SalienceTree *tree, int p, int q){ 
  tree->node[q].parent = p;
  tree->node[p].area +=  tree->node[q].area;
  tree->node[p].sumPix += tree->node[q].sumPix;
  tree->node[p].minPix = MIN(tree->node[p].minPix,tree->node[q].minPix);
  tree->node[p].maxPix = MAX(tree->node[p].maxPix,tree->node[q].maxPix);
  
}

void Phase1(SalienceTree *tree, EdgeQueue *queue, Pixel *img, 
	    int width, int height, double (*salience)(Pixel, Pixel)){
  /* pre: tree has been created with imgsize= width*height 
          queue initialized accordingly;
   */
  int imgsize = width*height;
  int p,x,y;

 
  MakeSet(tree, img, 0);
  
  for (x=1;x<width;x++){
    MakeSet(tree, img, x);
    if (img[x]==img[x-1]){
      Union(tree,x,x-1);
    } else {
      EdgeQueuePush(queue,x,x-1,(*salience)(img[x],img[x-1]));
    }
  }

  for (y = 1; y<height; y++){
    p=y*width;
    MakeSet(tree, img, p);
    if (img[p]==img[p-width]){
      Union(tree,p, p-width);
    } else {
      EdgeQueuePush(queue,p, p-width,(*salience)(img[p],img[p-width]));
    }
    p++;  
    for (x=1;x<width;x++,p++){
      MakeSet(tree, img, p);
      if (img[p]==img[p-width]){
	Union(tree,p, p-width);
      } else {
	EdgeQueuePush(queue,p, p-width,(*salience)(img[p],img[p-width]));
      }
      if (img[p]==img[p-1]){
	Union(tree,p, p-1);
      } else {
	EdgeQueuePush(queue,p, p-1,(*salience)(img[p],img[p-1]));
      }
      
    }    

  }
  
}


void GetAncestors(SalienceTree *tree, int *p, int*q){
  int pp, pq, temp;
  *p = LevelRoot(tree, *p);
  *q = LevelRoot(tree, *q);
  pp = tree->node[*p].parent;
  pq = tree->node[*q].parent;
  if (*p<*q){
    temp = *p;
    *p=*q;
    *q=temp;
    temp = pp;
    pp=pq;
    pq=temp;
  }
  while ((*p!=*q) && (pp!=BOTTOM) &&(pq!=BOTTOM)){
    *q=Par(tree,*q);
    if (*p<*q){
      temp = *p;
      *p=*q;
      *q=temp;
    }
    pp = tree->node[*p].parent;
    pq = tree->node[*q].parent;
  }
  if (pp==BOTTOM) {
    *q=FindRoot2(tree,*q);
  } else if (pq==BOTTOM){
    *p=FindRoot2(tree,*p);
  }
}

void Phase2(SalienceTree *tree, EdgeQueue *queue, 
	    Pixel *img, 
	    int width, int height){
  Edge *currentEdge;
  int v1, v2, temp, r;
  double oldalpha=0,alpha12; 
  while (!IsEmpty(queue)){
    currentEdge = EdgeQueueFront(queue);
    v1 =currentEdge->p;
    v2 =currentEdge->q;
    GetAncestors(tree,&v1,&v2);
    alpha12 = currentEdge->alpha;
    assert(oldalpha<=alpha12);
    EdgeQueuePop(queue);
    if (v1!=v2) {
      if (v1 < v2) {
	temp=v1;
	v1=v2;
	v2=temp;
      }
      if (tree->node[v1].alpha < alpha12){
	r =NewSalienceNode(tree, alpha12);
	Union2(tree,r,v1);
	Union2(tree,r,v2);
      } else {
	Union2(tree,v1,v2);
      }
    }
    oldalpha=alpha12;
  }
}

SalienceTree *MakeSalienceTree(Pixel *img, int width, int height){
  int imgsize = width*height;
  EdgeQueue *queue = EdgeQueueCreate((CONNECTIVITY/2)*imgsize);
  SalienceTree *tree;
  tree = CreateSalienceTree(imgsize);
  assert(tree!=NULL);
  assert(tree->node!=NULL);
  fprintf(stderr,"Phase1 started\n");
  Phase1(tree,queue,img,width,height,simpleSalience);
  fprintf(stderr,"Phase2 started\n");
  Phase2(tree,queue,img,width,height);
  fprintf(stderr,"Phase2 done\n");
  EdgeQueueDelete(queue);
  return tree;
}


void SalienceTreeAreaFilter(SalienceTree *tree, Pixel *out, int lambda){
  int i, imgsize = tree->maxSize/2;
  if (lambda <= imgsize){
    tree->node[tree->curSize-1].outval =
      tree->node[tree->curSize-1].sumPix/tree->node[tree->curSize-1].area;
    for (i = tree->curSize-2; i>=0; i--){
      
      if (IsLevelRoot(tree,i) && (tree->node[i].area>=lambda)){
	tree->node[i].outval = tree->node[i].sumPix/tree->node[i].area;
      } else {
	tree->node[i].outval = tree->node[tree->node[i].parent].outval;
      }
    }
  } else {
    for (i = tree->curSize-1; i>=0; i--){
      tree->node[i].outval = 0;
    }
  }
  for (i = 0; i < imgsize; i++)
    out[i] = tree->node[i].outval;
}

#define NodeSalience(tree, p) (tree->node[Par(tree,p)].alpha)

void SalienceTreeSalienceFilter(SalienceTree *tree, Pixel *out, double lambda){
  int i, imgsize = tree->maxSize/2;
  if (lambda <= tree->node[tree->curSize-1].alpha){
    tree->node[tree->curSize-1].outval =
      tree->node[tree->curSize-1].sumPix/tree->node[tree->curSize-1].area;
    for (i = tree->curSize-2; i>=0; i--){
      
      if (IsLevelRoot(tree,i) && (NodeSalience(tree,i)>=lambda)){
	tree->node[i].outval = tree->node[i].sumPix/tree->node[i].area;
      } else {
	tree->node[i].outval = tree->node[tree->node[i].parent].outval;
      }
    }
  } else {
    for (i = tree->curSize-1; i>=0; i--){
      tree->node[i].outval = 0;
    }
  }
  for (i = 0; i < imgsize; i++)
    out[i] = tree->node[i].outval;
}


short ImagePGMAsciiRead(char *fname)
{
   FILE *infile;
   ulong i;
   int c;

   infile = fopen(fname, "r");
   if (infile==NULL) {
      fprintf (stderr, "Error: Can't read the ASCII file: %s !", fname);
      return(0);
   }
   fscanf(infile, "P2\n");
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
      fscanf(infile, "%d", &c);
      gval[i] = c;
   }
   fclose(infile);
   return(1);
} /* ImagePGMAsciiRead */


short ImagePGMBinRead(char *fname)
{
   FILE *infile;
   int c, i;
   ubyte *buf = NULL;

   infile = fopen(fname, "rb");
   if (infile==NULL) {
      fprintf (stderr, "Error: Can't read the binary file: %s !", fname);
      return(0);
   }
   fscanf(infile, "P5\n");
   while ((c=fgetc(infile)) == '#')
      while ((c=fgetc(infile)) != '\n');
   ungetc(c, infile);
   fscanf(infile, "%d %d\n255\n", &width, &height);
   size = width*height;

   buf = malloc(size);
   if (buf==NULL) {
     fprintf (stderr, "Out of memory!");
     fclose(infile);
     return(0);
   } 
   gval = malloc(size*sizeof(Pixel));
   if (gval==NULL) {
     fprintf (stderr, "Out of memory!");
     fclose(infile);
     return(0);
   } 
   fread(buf, 1, size, infile);

   for (i=0; i<size; i++)
     gval[i]=buf[i];
   free(buf);
   fclose(infile);
   return(1);
} /* ImagePGMBinRead */


short ImagePGMRead(char *fname)
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
   if (strcmp(id, "P2")==0) return(ImagePGMAsciiRead(fname));
   else if (strcmp(id, "P5")==0) return(ImagePGMBinRead(fname));
   else {
     fprintf (stderr, "Unknown type of the image!");
     return(0);
   }
} /* ImagePGMRead */

int ImagePGMBinWrite(char *fname)
{
   FILE *outfile;
   ubyte *buf=NULL;
   int i;

   outfile = fopen(fname, "wb");
   if (outfile==NULL) {
      fprintf (stderr, "Error: Can't write the image: %s !", fname);
      return(-1);
   }
   fprintf(outfile, "P5\n%d %d\n255\n", width, height);

   buf = malloc(size);
   if (buf==NULL) {
     fprintf (stderr, "Out of memory!");
     fclose(outfile);
     return(-1);
   } 
   for (i=0;i<size;i++)
     buf[i]=out[i];
   fwrite(buf, 1, (size_t)(size), outfile);
   free(buf);
   fclose(outfile);
   return(0);
} /* ImagePGMBinWrite */


int main (int argc, char *argv[]) {
  
   char *imgfname, *outfname = "out.pgm";
   int r;
   ulong i;
   clock_t start;
   struct tms tstruct;
   long tickspersec = sysconf(_SC_CLK_TCK);  
   float musec;
   SalienceTree *tree;

   if (argc<3)
   {
      printf("Usage: %s <input image> <lambda>  [omegafactor] [output image] \n", argv[0]);
      exit(0);
   }

   imgfname = argv[1];

   lambda = atoi(argv[2]);
   if (argc>3)  omegafactor = atof(argv[3]);

   if (argc>4)  outfname = argv[4];
 
   
   if (!ImagePGMRead(imgfname)) 
     return(-1);

   out = malloc(size*sizeof(Pixel));

   printf("Filtering image '%s' using attribute area with lambda=%d\n", imgfname, lambda);
   printf("Image: Width=%d Height=%d\n", width, height);
   
   printf("Data read, start filtering.\n");
   start = times(&tstruct);
   
   tree = MakeSalienceTree(gval,width,height);

  musec = (float)(times(&tstruct) - start)/((float)tickspersec);

   printf("wall-clock time: %f s\n",musec);
   SalienceTreeAreaFilter(tree,out,lambda);
   /*SalienceTreeSalienceFilter(tree,out,(double)lambda);
   */
   musec = (float)(times(&tstruct) - start)/((float)tickspersec);

   printf("wall-clock time: %f s\n",musec);
   
   r = ImagePGMBinWrite(outfname);
   free(out);  
   if (r)  printf("Filtered image written to '%s'\n", outfname);
   
   free(gval);
   return(0);
} /* main */
