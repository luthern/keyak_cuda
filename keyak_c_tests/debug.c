#include <stdio.h>
#define debug() ___debug(__FILE__,__LINE__)

void ___debug(const char * file, int line)
{
    fprintf(stderr, "%s: %d\n",file,line);
}
