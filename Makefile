# Compiler
# Compiler to use (change to 'clang' or other if necessary)
CC = gcc 
FLAGS = -g -Wall -Wextra
HEADERS = headers
SOURCE = src

all: final

final: main.o merge_sort.o read_csv.o
	echo "Linking and producing the final executable"
	gcc $(FLAGS) main.o merge_sort.o read_csv.o -o final

main.o: main.c 
	echo "Compiling main.c"
	$(CC) $(FLAGS) -c main.c 

merge_sort.o: $(SOURCE)/merge_sort.c 
	echo "Compiling merge_sort.c"
	$(CC) $(FLAGS) -c $(SOURCE)/merge_sort.c 

read_csv.o: $(SOURCE)/read_csv.c 
	echo "Compiling read_csv.c"
	$(CC) $(FLAGS) -c $(SOURCE)/read_csv.c 

clean: 
	echo "Removing everything but the source file"
	rm *.o
