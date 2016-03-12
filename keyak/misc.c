
#include "misc.h"

#include <time.h>

static const char * _msg;
struct timespec tstart={0,0}, tend={0,0};

void timer_start(const char * msg)
{
    _msg = msg;
    clock_gettime(CLOCK_MONOTONIC, &tstart);
}

void timer_end()
{
    clock_gettime(CLOCK_MONOTONIC, &tend);
    printf("%s time: %.1f ns\n", _msg,
            ((double)tend.tv_sec * 1.0e-9 + tend.tv_nsec) - 
            ((double)tstart.tv_sec * 1.0e-9 + tstart.tv_nsec));
}
