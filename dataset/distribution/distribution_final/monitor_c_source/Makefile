# Makefile for IBEA

# Compiler
CC = gcc

# Compiler options
CFLAGS = -g -Wall

# all object files
SEL_OBJECTS = monitor_internal.o monitor_user.o monitor.o

monitor : $(SEL_OBJECTS)
	$(CC) $(CFLAGS) -lm $(SEL_OBJECTS) -o monitor

monitor_internal.o : monitor_internal.c monitor.h
	$(CC) $(CFLAGS) -c monitor_internal.c 

monitor_user.o : monitor_user.c monitor.h
	$(CC) $(CFLAGS) -c monitor_user.c

monitor.o : monitor.c monitor.h
	$(CC) $(CFLAGS) -c monitor.c

clean:
	rm -f *~ *.o

