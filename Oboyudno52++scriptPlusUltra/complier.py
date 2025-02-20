import sys
import os

file_path = sys.argv[1]

if file_path[-2:] != ".🤙":
    print("BAD FILE BRO")
    exit()
with open(file_path, "r") as f: 
    lines = f.read().split()
    no_comments = []
    is_comment = False
    for op in lines:
        if op == ";":
            is_comment = not is_comment
            continue
        if is_comment == False:
            no_comments.append(op)

def compile_(code):
    program= """extern int putchar(int);
extern char getchar();

char array[30000];

int idx = 0;

int main (int arc, char *argv[]) {
"""
    for op in code:
        match op:
            case "52":
                program += "    idx += 1; \n"
            case "69":
                program += "    idx -= 1; \n"
            case "oboyudno":
                program += "    array[idx]+=1; \n"
            case "kolenval":
                program += "    array[idx]-=1; \n"
            case "🤙":
                program += "    putchar(array[idx]); \n"
            case "1488":
                program += "    array[idx] = getchar(); \n"
            case "dom":
                program += "    while(array[idx]){ \n"
            case "chai":
                program += "    } \n"
            case _:
                print("BAD TOKEN ", op)
    program += "}"
    with open("./tmp.c", "w") as f: 
        f.write(program)
    os.system("gcc -Wall tmp.c -o tmp")
    os.system("./tmp")
    os.remove("./tmp.c")
    os.remove("./tmp")
    
    
compile_(no_comments)

# python ./complier.py ./HelloWorld.🤙
# python ./complier.py ./Fibbonachi.🤙
# python ./complier.py ./Summ.🤙