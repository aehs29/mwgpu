/* Includes */

#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>


/* Structures */

typedef struct nodes_struct
{
	int count=0;
	int dimensions=0;
    int fixed_count=0;
    float center_x=0;
    float center_y=0;
    float center_z=0;

} nodes_struct;

typedef struct elem_struct
{
	int count=0;
	int nodes=0;

} elem_stuct;
