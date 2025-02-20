extern int putchar(int);
extern char getchar();

char array[30000];

int idx = 0;

int main (int arc, char *argv[]) {
    array[idx] = getchar(); 
    idx += 1; 
    array[idx] = getchar(); 
    while(array[idx]){ 
    idx -= 1; 
    array[idx]+=1; 
    idx += 1; 
    array[idx]-=1; 
    } 
    idx -= 1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    array[idx]-=1; 
    putchar(array[idx]); 
}