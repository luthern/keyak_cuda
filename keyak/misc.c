
#include "misc.h"

#include <time.h>

static const char * _msg;
struct timespec tstart={0,0}, tend={0,0};

static int acum = 0;
static double total = 0;

void timer_start(const char * msg)
{
    _msg = msg;
    clock_gettime(CLOCK_MONOTONIC, &tstart);
}


void timer_accum()
{
    acum = 1;
    clock_gettime(CLOCK_MONOTONIC, &tend);
    total += (((double)tend.tv_sec * 1.0e-9 + tend.tv_nsec) -
            ((double)tstart.tv_sec * 1.0e-9 + tstart.tv_nsec));

}

void timer_end()
{
    if (acum)
    {
    }
    else
    {
        total = (((double)tend.tv_sec * 1.0e-9 + tend.tv_nsec) -
                ((double)tstart.tv_sec * 1.0e-9 + tstart.tv_nsec));
    }
    acum = 0;
    printf("%s time: %.1f ns\n", _msg, total);
}
