CC ?= clang
CFLAGS = -Ofast -g
LDFLAGS =
LDLIBS = -lepoxy -lm
INCLUDES = glblas.c

TARGETS = sasum saxpy scopy sdot sgemm sgemm4x4 sscal sswap

all: $(TARGETS)

sasum: demos/sasum.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) -o $@

saxpy: demos/saxpy.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) -o $@

scopy: demos/scopy.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) -o $@

sdot: demos/sdot.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) -o $@

sgemm: demos/sgemm.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) -o $@

sgemm4x4: demos/sgemm4x4.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) -o $@

sscal: demos/sscal.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) -o $@

sswap: demos/sswap.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) -o $@

clean:
	rm -f sasum saxpy scopy sdot sgemm sgemm4x4 sscal sswap