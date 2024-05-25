# Makefile for COMP4300/8300 Assignment 2 (OpenMP)
.SUFFIXES:
.PRECIOUS: %.o

# Define variables
HDRS = serAdvect.h parAdvect.h
OBJS = serAdvect.o parAdvect.o
PROG = testAdvect
CC = gcc
CFLAGS = -Wall -Werror -O3 -fopenmp
LDFLAGS = -lgomp -lm -lpapi

# Default target
all: $(PROG)

# Linking the program
$(PROG): $(OBJS) $(PROG).o
	$(CC) -o $@ $(PROG).o $(OBJS) $(LDFLAGS)

# Compiling object files
%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -c $<

# Clean up
clean:
	rm -f *.o $(PROG)
