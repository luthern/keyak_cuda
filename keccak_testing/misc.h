#ifndef _MISC_H_
#define _MISC_H_

#include <stdio.h>
#include <time.h>
#include <stdint.h>

#define MAX(a,b)        ((a) > (b) ? (a) : (b))
#define MIN(a,b)        ((a) < (b) ? (a) : (b))
#define CEIL(x,y)       (((x) + (y) - 1) / (y))

#define debug()         (printf("%s: %d\n", __FILE__, __LINE__))

struct timer
{
    char msg[100];
    struct timespec tstart,tend;
    double total;
    int accum;
};

void timer_start(struct timer * t, const char * msg);
void timer_end(struct timer * t );
void timer_accum(struct timer * t );

void motorist_timers_end();

void dump_hex(uint8_t * buf, int len);

#endif
