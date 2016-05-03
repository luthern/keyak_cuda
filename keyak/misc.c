
#include "misc.h"
#include <string.h>


void _dump_hex(uint8_t * buf, int len, int nl)
{
#ifdef USE_TIMERS
    while(len--)
        printf("%02hhx", *buf++);
    if (nl) printf("\n");
#endif
}

void timer_start(struct timer * t, const char * msg)
{
#ifdef USE_TIMERS
    if (msg[0] != t->msg[0]) memmove(t->msg, msg, strlen(msg)+1);
    clock_gettime(CLOCK_MONOTONIC, &t->tstart);
#endif
}


void timer_accum(struct timer * t)
{
#ifdef USE_TIMERS
    t->accum = 1;
    clock_gettime(CLOCK_MONOTONIC, &t->tend);
    t->total += (((double)t->tend.tv_sec + 1.0e-9 * (double)t->tend.tv_nsec) -
            ((double)t->tstart.tv_sec + 1.0e-9 * (double)t->tstart.tv_nsec));

#endif
}

void timer_end(struct timer * t)
{
#ifdef USE_TIMERS
    if (t->accum)
    {
    }
    else
    {
        clock_gettime(CLOCK_MONOTONIC, &t->tend);
        t->total = (((double)t->tend.tv_sec + 1.0e-9 * (double)t->tend.tv_nsec) -
                ((double)t->tstart.tv_sec + 1.0e-9 * (double)t->tstart.tv_nsec));
    }
    t->accum = 0;
    fprintf(stderr,"%s time: %.5f s\n", t->msg, t->total);
#endif
}

