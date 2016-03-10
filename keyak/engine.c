#include <string.h>
#include "engine.h"
#include "piston.h"

void engine_init(Engine * e, Piston * pistons)
{
    memset(e,0,sizeof(Engine));
    e->pistons = pistons;
    e->phase = EngineFresh;
}
