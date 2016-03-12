
#include "misc.h"
#include <string.h>


void timer_start(struct timer * t, const char * msg)
{
    if (msg[0] != t->msg[0]) memmove(t->msg, msg, strlen(msg)+1);
    clock_gettime(CLOCK_MONOTONIC, &t->tstart);
}


void timer_accum(struct timer * t)
{
    t->accum = 1;
    clock_gettime(CLOCK_MONOTONIC, &t->tend);
    t->total += (((double)t->tend.tv_sec + 1.0e-9 * (double)t->tend.tv_nsec) -
            ((double)t->tstart.tv_sec + 1.0e-9 * (double)t->tstart.tv_nsec));

}

void timer_end(struct timer * t)
{
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
    printf("%s time: %.4f s\n", t->msg, t->total);
}
