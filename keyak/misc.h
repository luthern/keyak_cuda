#ifndef _MISC_H_
#define _MISC_H_

#include <stdio.h>

#define MAX(a,b)        ((a) > (b) ? (a) : (b))
#define CEIL(x,y)       (((x) + (y) - 1) / (y))

#define debug()         (printf("%s: %d\n", __FILE__, __LINE__))

void timer_start(const char * msg);
void timer_end();
void timer_accum();

#endif
