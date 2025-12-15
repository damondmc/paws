from pathlib import Path

def write_search_subfile(filename, executable_path, transfer_executable, output_path, 
                   error_path, log_path, arg_list_string, 
                   accounting_group, user, 
                   request_memory='15GB', request_disk='3GB', request_cpu=1, 
                   use_osg=False, use_osdf=False, image=None):
    """
    Writes the Condor submission (.sub) file.

    Parameters:
        - subFileName (str): Path to the .sub file to be created.
        - executablePath (str): Path to the executable (e.g., lalapps_Weave).
        - transfer_executable (bool): Whether HTCondor should transfer the executable to the worker node.
        - outputPath (str): Path for standard output stream.
        - errorPath (str): Path for standard error stream.
        - logPath (str): Path for the HTCondor log file.
        - argListString (str): The arguments to pass to the executable.
        - accounting_group (str): LIGO accounting tag (e.g., ligo.prod.o4...).
        - user (str): The username for accounting.
        - request_memory (str): Amount of memory to request (default: '15GB').
        - request_disk (str): Amount of disk space to request (default: '3GB').
        - request_cpu (int): Number of CPUs to request (default: 1).
        - OSG (bool): If True, configures for Open Science Grid (file transfer enabled).
        - OSDF (bool): If True, configures for Open Science Data Federation (scitokens).
        - image (str, optional): Path to Singularity image if required.
    """
    # Ensure directory exists
    Path(Path(filename).resolve().parent).mkdir(parents=True, exist_ok=True)
    
    with open(filename, 'w') as subfile:
        subfile.write('universe = vanilla\n')
        subfile.write('notification = Never\n')
        subfile.write('request_memory = {0}\n'.format(request_memory)) 
        subfile.write('request_disk = {0}\n'.format(request_disk))   
        subfile.write('request_cpus = {0}\n'.format(request_cpu))
        subfile.write('getenv = True\n')
        
        subfile.write('accounting_group = {0}\n'.format(accounting_group))
        subfile.write('accounting_group_user = {0}\n\n'.format(user))
        
        if image is not None:
            subfile.write('MY.SingularityImage = "{}"\n\n'.format(image))
            
        subfile.write('output = {0}\n'.format(output_path))
        subfile.write('error = {0}\n'.format(error_path))
        subfile.write('log = {0}\n'.format(log_path))
        subfile.write('max_retries = {0}\n'.format(2)) # Retry if non-zero exit code
        subfile.write('periodic_release = (HoldReasonSubCode == 13)\n') # Release if transfer failed
        subfile.write('executable = {0}\n'.format(executable_path))

        if use_osg == False: 
            # Local cluster execution
            subfile.write("arguments = $(argList)\n\n")   
        else: 
            # OSG Execution
            subfile.write('arguments = {0}\n\n'.format(arg_list_string))
            subfile.write('stream_output = True\n')
            subfile.write('stream_error = True\n\n')
            subfile.write('should_transfer_files = YES\n')
            subfile.write('when_to_transfer_output = ON_SUCCESS\n') 
            subfile.write('success_exit_code = 0\n')
            subfile.write('transfer_executable={0}\n'.format(str(transfer_executable)))
            subfile.write('transfer_input_files = $(TRANSFERFILES)\n')
            subfile.write('transfer_output_files = $(OUTPUTFILE)\n')
            subfile.write('transfer_output_remaps = "$(OUTPUTFILE)=$(REMAPOUTPUTFILE)"\n\n')
            if use_osdf:
                subfile.write('use_oauth_services = scitokens\n') 
            subfile.write('queue 1')

    return 0

def write_search_dagfile(file_name, job_name, sub_file_name, job_num, arg_list_string):
    """
    Appends a JOB entry to the Condor DAG file.

    Parameters:
        - dagFileName (str): Path to the .dag file.
        - jobName (str): Base name for the job (e.g., 'Target_Search').
        - subFileName (str): Path to the corresponding .sub file.
        - jobNum (int): Index of the current job (used for JobID).
        - argListString (str): Arguments specific to this node (VARS).
    """
    Path(Path(file_name).resolve().parent).mkdir(parents=True, exist_ok=True)
    with open(file_name, 'a') as dagfile: # 'a' mode appends to the file
        dagfile.write('JOB {0}_{1} {2}\n'.format(job_name, job_num, sub_file_name))
        dagfile.write('VARS {0}_{1} JobID="{1}" {2}'.format(job_name, job_num, arg_list_string))
        dagfile.write('\n')
    return 0