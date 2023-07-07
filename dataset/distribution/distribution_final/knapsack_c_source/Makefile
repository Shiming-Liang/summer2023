# Makefile for Variator

# Compiler
CC = gcc

# Compiler options
CFLAGS = -g -Wall -pedantic

# all object files
SEL_OBJECTS = variator_user.o variator.o variator_internal.o

knapsack : $(SEL_OBJECTS)
	$(CC) $(CFLAGS) -lm $(SEL_OBJECTS) -o knapsack

variator_internal.o : variator_internal.c variator_internal.h variator.h variator_user.h
	$(CC) $(CFLAGS) -c variator_internal.c 

variator_user.o : variator_user.c variator_user.h variator.h
	$(CC) $(CFLAGS) -c variator_user.c

variator.o : variator.c variator.h variator_user.h variator_internal.h
	$(CC) $(CFLAGS) -c variator.c

clean:
	rm -f *~ *.o
