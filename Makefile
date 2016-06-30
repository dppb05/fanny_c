CC=gcc
CFLAGS=-Wall -c
EXEDIR=.

all: $(EXEDIR)/fanny

$(EXEDIR)/fanny: stex.o matrix.o util.o fanny.o
	$(CC) stex.o matrix.o util.o fanny.o -o $@ -lm

fanny.o: fanny.c
	$(CC) $(CFLAGS) fanny.c

util.o: util.c
	$(CC) $(CFLAGS) util.c

matrix.o: matrix.c
	$(CC) $(CFLAGS) matrix.c

stex.o: stex.c
	$(CC) $(CFLAGS) stex.c

clean:
	rm -rf *o $(EXEDIR)/fanny
