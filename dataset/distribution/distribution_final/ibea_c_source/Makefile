# Makefile for IBEA

# Compiler
CC = gcc

# Compiler options
CFLAGS = -g -Wall

# all object files
SEL_OBJECTS = ibea_functions.o ibea_io.o ibea.o

ibea : $(SEL_OBJECTS)
	$(CC) $(CFLAGS) -lm $(SEL_OBJECTS) -o ibea

ibea_functions.o : ibea_functions.c ibea.h
	$(CC) $(CFLAGS) -c ibea_functions.c 

ibea_io.o : ibea_io.c ibea.h
	$(CC) $(CFLAGS) -c ibea_io.c

ibea.o : ibea.c ibea.h
	$(CC) $(CFLAGS) -c ibea.c

clean:
	rm -f *~ *.o

