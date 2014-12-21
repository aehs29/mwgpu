#include "file_handling.h"

using namespace std;

int load_nodefile(const char * nodefile_name, float *& nodes , nodes_struct * ns){

	ifstream node_file(nodefile_name);
	string line;
	
	int line_count=0;
	int node_init_index=0;
	
	while(std::getline(node_file,line))
	{
        // Get rid of carriage return
		if(line[line.length()-1]=='\r')
			line[line.length()-1]='\0';
		std::stringstream  lineStream(line);
        std::string        cell;
		int val_count=0;
        while(std::getline(lineStream,cell,' '))
        {
			if(cell.length()>0)
			{
				if(line_count==0)
				{
					if(val_count==0)
						// Get Number of Nodes
						ns->count=atoi(cell.c_str());
					else if(val_count==1){
						//Get Number of Dimensions
						ns->dimensions=atoi(cell.c_str());
						nodes=new float[ns->count*ns->dimensions];
						// nodes_orig=new float[node_count*node_dimensions];

					}

				}
				else
				{
					// Check speed of conversion
					char* p;
					float converted = strtof(cell.c_str(), &p);
					if (*p) 
					{
						// conversion failed because the input wasn't a number
					}
					else 
					{
						if(line_count==1 & val_count==0)
							node_init_index=converted;
						if(val_count>0)
							nodes[ns->dimensions*(line_count-1)+(val_count-1)]=converted;
					}
				}
				val_count++;
			}
		}
		line_count++;
	}

	// Get centers
	// use modulo operator
	int i=0;
	float tmpminX=0;
	float tmpmaxX=0;
	float tmpminY=0;
	float tmpmaxY=0;
	float tmpminZ=0;
	float tmpmaxZ=0;
	for (i=0;i<ns->count*ns->dimensions;i++){
		switch (i%3){
		case 0:
			if(nodes[i]<tmpminX)
				tmpminX=nodes[i];
			if(nodes[i]>tmpmaxX)
				tmpmaxX=nodes[i];
			break;
		case 1:
			if(nodes[i]<tmpminY)
				tmpminY=nodes[i];
			if(nodes[i]>tmpmaxY)
				tmpmaxY=nodes[i];
			break;
		case 2:
			if(nodes[i]<tmpminZ)
				tmpminZ=nodes[i];
			if(nodes[i]>tmpmaxZ)
				tmpmaxZ=nodes[i];
			break;
		}
		ns->center_x=(tmpmaxX-tmpminX)/2;
		ns->center_y=(tmpmaxY-tmpminY)/2;
		ns->center_z=(tmpmaxZ-tmpminZ)/2;
	}

	return node_init_index;
}


void load_elemfile(const char * elemfile_name, GLushort *& elem , elem_struct * es, int node_init_index){

ifstream elem_file(elemfile_name);	
	int line_count=0;
	string line;
	int node_start=0;
	while(std::getline(elem_file,line))
	{
        // Get rid of carriage return
		if(line[line.length()-1]=='\r')
			line[line.length()-1]='\0';
		std::stringstream  lineStream(line);
        std::string        cell;
		int val_count=0;
        while(std::getline(lineStream,cell,' '))
        {
			if(cell.length()>0)
			{
				if(line_count==0)
				{
					if(val_count==0)
						// Get Number of Elements
						es->count=atoi(cell.c_str());
					else if(val_count==1){
						//Get Number of Element Nodes
						es->nodes=atoi(cell.c_str());
						// Allocate array
						elem=new GLushort[es->count*es->nodes];
					}
				}
				else
				{
					char* p;
					GLshort converted = strtol(cell.c_str(), &p,10); // Add base 10
					if (*p) 
					{
						// conversion failed because the input wasn't a number
					}
					else 
					{
						if(val_count>0)
						{
							elem[es->nodes*(line_count-1)+(val_count-1)]=converted-node_init_index;
						}
					}
				}
				val_count++;
			}
		}
		line_count++;
	}
}

int load_eigenvalsfile(const char * evalsfile_name, float *& eigenVals){
    
    // No need for columns in EIGENVALUES
    // Get file handle
	std::ifstream csv_file_rows(evalsfile_name);
	std::ifstream csv_file(evalsfile_name);
	unsigned int row_count=0;
	unsigned int tot_rowCount;
	string line;

    // Read whole file to get row and count - Don't like it but its only done once
	while (std::getline(csv_file_rows,line)){
		row_count++;
	}

	tot_rowCount=row_count;

	// Allocate array
	eigenVals=new float[tot_rowCount];

	row_count=0;				// reset value

	// Read file to initialize array
	while (std::getline(csv_file,line)){
		// Get rid of carriage return
		if(line[line.length()-1]=='\r')
			line[line.length()-1]='\0';
		std::stringstream lineStream(line);
		std::string colVal;
		
		// Separate values by ',' (CSV)
		while(std::getline(lineStream,colVal,',')){
			if(colVal.length()>0){ // Could assert also

				// Convert char* to float
				char *p;
				float converted = strtof(colVal.c_str(),&p);
				if (*p){}
				else{
					eigenVals[row_count]=converted;
				}
			}
		}
		row_count++;
	}

	return tot_rowCount;		// # of EigenValues

}

void load_fxdnodesfile(const char * fxdnfile_name, int *& fxdnodes , nodes_struct * ns){

	// No need for columns in Fixed Nodes either
    // Get file handle
	std::ifstream fixed_nodes_rows(fxdnfile_name);
	std::ifstream fixed_nodes_file(fxdnfile_name);

	string line;

    int row_count=0;				// Reset

    // Read whole file to get row and count
	while (std::getline(fixed_nodes_rows,line)){
		row_count++;
	}

	ns->fixed_count=row_count;

	// Allocate array
	fxdnodes=new int[ns->fixed_count];

	row_count=0;				// reset value

	// Read file to initialize array
	while (std::getline(fixed_nodes_file,line)){

		// Get rid of carriage return
		if(line[line.length()-1]=='\r')
			line[line.length()-1]='\0';
		std::stringstream lineStream(line);
		std::string colVal;

		// Separate values by ',' (CSV)
		while(std::getline(lineStream,colVal,',')){
			if(colVal.length()>0){ // Could assert also
				// Convert char* to float
				char *p;
				int converted = strtol(colVal.c_str(),&p,10);
				if (*p){}
				else{
					fxdnodes[row_count]=converted-1; // These are in MATLAB index syntax hence the -1
				}
			}
		}
		row_count++;
	}

}

void load_eigenvecfile(const char * evecfile_name, float *& eigenVecs , nodes_struct * ns, int eigencount){

 // Get file handle

	std::ifstream eigenvec_file_rows(evecfile_name);
	std::ifstream eigenvec_file(evecfile_name);

	string line;
	// Columns should be same as eigenVals count
	// Rows should be the same as nodes x dimensions
	unsigned int col_countEvec=0;
	// unsigned int tot_colCountEvec;
	
	// Columns same as EigenVals count
	// tot_colCountEvec=eigencount;

	// Allocate array
	eigenVecs=new float[(ns->count-ns->fixed_count)*ns->dimensions*eigencount];

	int row_count=0;				// reset value

	// Read file to initialize array
	while (std::getline(eigenvec_file,line)){

		// Get rid of carriage return
		if(line[line.length()-1]=='\r')
			line[line.length()-1]='\0';
		std::stringstream lineStream(line);
		std::string colVal;
		
		// Reset col value
		col_countEvec=0;

		// Separate values by ',' (CSV)
		while(std::getline(lineStream,colVal,',')){
			if(colVal.length()>0){ // Could assert also
				// Convert char* to float
				char *p;
				float converted = strtof(colVal.c_str(),&p);
				if (*p){}
				else{
					eigenVecs[(eigencount*row_count)+col_countEvec]=converted;
				}
			}
			col_countEvec++;
		}
		row_count++;
	}



}

void load_Psifile(const char * psifile_name, float *& Psi , nodes_struct * ns, int eigencount){


	std::ifstream psi_file(psifile_name);
	string line;
	// Columns should be same as eigenVals count
	// Rows should be the same as nodes x dimensions
    int col_countEvec=0;


	// Allocate array
	Psi=new float[(ns->count-ns->fixed_count)*ns->dimensions*eigencount];

	int row_count=0;				// reset value

	// Read file to initialize array
	while (std::getline(psi_file,line)){

		// Get rid of carriage return
		if(line[line.length()-1]=='\r')
			line[line.length()-1]='\0';
		std::stringstream lineStream(line);
		std::string colVal;
		
		// Reset col value
		col_countEvec=0;

		// Separate values by ',' (CSV)
		while(std::getline(lineStream,colVal,',')){
			if(colVal.length()>0){ // Could assert also
				// Convert char* to float
				char *p;
				float converted = strtof(colVal.c_str(),&p);
				if (*p){}
				else{
					Psi[(eigencount*row_count)+col_countEvec]=converted;
				}
			}
			col_countEvec++;
		}
		row_count++;
	}
}

void save_Obj(string filename, float *nodes, GLushort *elem, int node_count, int node_dimensions, int elem_count, int elem_nodes, int node_init_index){
	time_t t = time(0);   // get time now
	struct tm * now = localtime( & t ); // Convert to Localtime

	// Append to filename
	std::stringstream ss;
	ss<<"_"<<now->tm_min<<"_"<<now->tm_sec;
	filename.append(ss.str());
	filename.append(".obj");
	
	// ToDo Print info of the file
	ofstream objfile (filename.c_str());
	if (objfile.is_open())
	{
		// Output nodes
		for(int i=0;i<node_count;i++){
			objfile << "v ";
			for(int j=0;j<node_dimensions;j++){
				objfile << nodes[i*node_dimensions+j]<<" ";
			}
			objfile << std::endl;
		}
		objfile << std::endl;

		// Output elems
		for(int i=0;i<elem_count;i++){
			objfile << "f ";
			for(int j=0;j<elem_nodes;j++){
				objfile << elem[i*elem_nodes+j]+node_init_index<<" ";
			}
			objfile << std::endl;
		}		
		objfile.close();
	}
	else cout << "Unable to write obj file";

}

void log(std::string filename, vector<string> lines){

	std::ofstream logfile;
	logfile.open(filename.c_str(), std::ofstream::out | std::ofstream::app);
	if (logfile.is_open())
	{
		// logfile<<tm_mday<<"/"<<tm_mon+1<<"/"<<tm_year+1990<<";
		for(int i=0;i <lines.size();i++){
			logfile<<lines.at(i)<<std::endl;
		}
		logfile.close();
	}
	else cout << "Unable to write log";
	lines.clear();
}

void init_log(std::string filename){

	time_t t = time(0);   // get time now
	struct tm * now = localtime( & t ); // Convert to Localtime
	std::stringstream ss;
	ss<<now->tm_mday<<"/"<<now->tm_mon+1<<"/"<<now->tm_year+1990<<" -"<<now->tm_hour<<":"<<now->tm_min<<std::endl;
	
	std::ofstream logfile;
	logfile.open(filename.c_str(), std::ofstream::out | std::ofstream::app);
	if (logfile.is_open())
	{
		logfile<<std::endl<<std::endl;
		// logfile<<tm_mday<<"/"<<tm_mon+1<<"/"<<tm_year+1990<<std::endl;
		// logfile<<tm_min;
		logfile<<ss.str();
		logfile<<"Starting..."<<std::endl;

	}
	else cout << "Unable to write log";
}

void save_tetgen(string filename, float *nodes, GLushort *elem, int node_count, int node_dimensions, int elem_count, int elem_nodes){
	time_t t = time(0);   // get time now
	struct tm * now = localtime( & t ); // Convert to Localtime

	std::string elemfilename;
	elemfilename=filename;

	// Append to filename
	std::stringstream ss;
	ss<<"_"<<now->tm_min<<"_"<<now->tm_sec;
	filename.append(ss.str());
	filename.append(".node");
	elemfilename.append(ss.str());
	elemfilename.append(".ele");
	
	// ToDo Print info of the file
	ofstream nodefile (filename.c_str());
	if (nodefile.is_open())
	{
		nodefile<<node_count<<" "<<node_dimensions<<" 0 0"<<std::endl;
		// Output nodes
		for(int i=0;i<node_count;i++){
			nodefile << i<<" ";
			for(int j=0;j<node_dimensions;j++){
				nodefile << nodes[i*node_dimensions+j]<<" ";
			}
			nodefile << std::endl;
		}
		nodefile << std::endl;

		nodefile.close();

		// Output elems
		// for(int i=0;i<elem_count;i++){
		// 	objfile << "f ";
		// 	for(int j=0;j<elem_nodes;j++){
		// 		objfile << elem[i*elem_nodes+j]<<" ";
		// 	}
		// 	objfile << std::endl;
		// }		
		// objfile.close();
	}
	else cout << "Unable to write node file";

	ofstream elemfile (elemfilename.c_str());
	if (elemfile.is_open())
	{
		elemfile<<elem_count<<" "<<elem_nodes<<" 0"<<std::endl;
		// Output elems
		for(int i=0;i<elem_count;i++){
			elemfile << i<<" ";
			for(int j=0;j<elem_nodes;j++){
				elemfile << elem[i*elem_nodes+j]<<" ";
			}
			elemfile << std::endl;
		}
		elemfile << std::endl;
		elemfile.close();

	}
	else cout << "Unable to write ele file";

}
