# === CONFIG ===
CC = mpicc
FLAGS = -g -Wall -Wextra
HEADERS = headers
SOURCE = src
EXEC = final

# === FIND ALL .c FILES RECURSIVELY ===
SRC_FILES := $(shell find $(SOURCE) -name '*.c')
OBJ_FILES := $(patsubst %.c,%.o,$(SRC_FILES))

# === RULES ===
all: $(EXEC)

$(EXEC): main.o $(OBJ_FILES)
	echo "Linking and producing the final executable"
	$(CC) $(FLAGS) $^ -o $@ -lm

# Compile each .c to .o, keeping folder structure
%.o: %.c
	echo "Compiling $<"
	$(CC) $(FLAGS) -I$(HEADERS) -I$(HEADERS)/tree -c $< -o $@

clean:
	echo "Cleaning up object files"
	find . -name '*.o' -delete
	# Do not remove the final executable
