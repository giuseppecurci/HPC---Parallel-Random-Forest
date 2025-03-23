# Compiler
# Compiler to use (change to 'clang' or other if necessary)
CC = gcc 
FLAGS = -g -Wall -Wextra
HEADERS = headers
SOURCE = src

all: final

final: main.o merge_sort.o read_csv.o utils.o
	echo "Linking and producing the final executable"
	$(CC) $(FLAGS) main.o merge_sort.o read_csv.o utils.o -o final 

%.o: $(SOURCE)/%.c
	echo "Compiling $<"
	$(CC) $(FLAGS) -c $< -o $@

clean: 
	echo "Removing everything but the final compiled file"
	rm *.o
