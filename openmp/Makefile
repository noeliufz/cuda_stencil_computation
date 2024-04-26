# Makefile for COMP4300/8300 Assignment 2 (OpenMP)
.SUFFIXES:
.PRECIOUS: %.o

HDRS=serAdvect.h parAdvect.h
OBJS=serAdvect.o parAdvect.o
PROG=testAdvect
CCFLAGS=-Wall -Werror -O3 -fopenmp

all: $(PROG) 

%: %.o $(OBJS)
	gcc -o $* $*.o $(OBJS) -lgomp -lm
%.o: %.c $(HDRS)
	gcc $(CCFLAGS) -c $*.c
clean:
	rm -f *.o $(PROG)
