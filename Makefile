CFLAG = -O2 -march=native -fno-strict-aliasing
CC = gcc
BIN = test-amxtile 
CFILES =test-amxtile.c 

all:
	$(CC) $(CFLAG) $(CFILES) -o $(BIN) $(LIBS)

clean:
	-rm $(BIN)

.PHONY: clean
