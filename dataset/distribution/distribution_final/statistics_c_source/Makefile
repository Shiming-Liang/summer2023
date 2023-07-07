CC=g++
CFLAGS=-c -Wall -pedantic
LDFLAGS=-lm
SOURCES=mann-whit.cc wilcoxon-sign.cc kruskal-wallis.cc dcdflib.cc fisher-indep.cc fisher-matched.cc
OBJECTS=$(SOURCES:.cc=.o)
EXECUTABLE=mann-whit kruskal-wallis wilcoxon-sign fisher-indep fisher-matched

all : $(EXECUTABLE)

kruskal-wallis : kruskal-wallis.o dcdflib.o
	$(CC) $(LDFLAGS) -o kruskal-wallis kruskal-wallis.o dcdflib.o

mann-whit : mann-whit.o dcdflib.o
	$(CC) $(LDFLAGS) -o mann-whit mann-whit.o dcdflib.o

wilcoxon-sign : wilcoxon-sign.o  dcdflib.o
	$(CC) $(LDFLAGS) -o wilcoxon-sign wilcoxon-sign.o dcdflib.o

fisher-indep : fisher-indep.o
	$(CC) $(LDFLAGS) -o fisher-indep fisher-indep.o

fisher-matched : fisher-matched.o
	$(CC) $(LDFLAGS) -o fisher-matched fisher-matched.o

.cc.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	@/bin/rm -f $(EXECUTABLE)	
	@/bin/rm -f *.o
	@/bin/rm -f *~
