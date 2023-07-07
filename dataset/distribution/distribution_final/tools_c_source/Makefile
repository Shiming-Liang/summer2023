CC=g++
CFLAGS=-c -Wall -pedantic
LDFLAGS=-lm
SOURCES=bound.cc filter.cc normalize.cc
OBJECTS=$(SOURCES:.cc=.o)
EXECUTABLE=bound filter normalize

all : $(EXECUTABLE)

bound : bound.o
	$(CC) $(LDFLAGS) -o bound bound.o

filter : filter.o
	$(CC) $(LDFLAGS) -o filter filter.o

normalize : normalize.o
	$(CC) $(LDFLAGS) -o normalize normalize.o

.cc.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	@/bin/rm -f $(EXECUTABLE)	
	@/bin/rm -f *.o
	@/bin/rm -f *~
