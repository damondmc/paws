from pathlib import Path


def write_search_subfile(
    filename,
    executable_path,
    transfer_executable,
    output_path,
    error_path,
    log_path,
    arg_list_string,
    accounting_group,
    user,
    request_memory="15GB",
    request_disk="3GB",
    request_cpu=1,
    use_osg=False,
    use_osdf=False,
    image=None,
):
    """
    Writes the Condor submission (.sub) file.
    """
    # Ensure directory exists
    Path(Path(filename).resolve().parent).mkdir(parents=True, exist_ok=True)

    with open(filename, "w") as subfile:
        if image is None:
            subfile.write("universe = vanilla\n")
        else:
            subfile.write("universe = container\n")
        subfile.write("notification = Never\n")
        subfile.write("request_memory = {0}\n".format(request_memory))
        subfile.write("request_disk = {0}\n".format(request_disk))
        subfile.write("request_cpus = {0}\n".format(request_cpu))
        subfile.write("getenv = False\n")

        subfile.write("accounting_group = {0}\n".format(accounting_group))
        subfile.write("accounting_group_user = {0}\n\n".format(user))

        if image is not None:
            subfile.write("container_image = {0}\n\n".format(image))

        subfile.write("output = {0}\n".format(output_path))
        subfile.write("error = {0}\n".format(error_path))
        subfile.write("log = {0}\n".format(log_path))
        # subfile.write("max_retries = {0}\n".format(2))
        # subfile.write("periodic_release = (HoldReasonSubCode == 13)\n")
        subfile.write("executable = {0}\n".format(executable_path))

        # Unified arguments line for both OSG and Local
        subfile.write("arguments = {0}\n\n".format(arg_list_string))

        if use_osg:
            # OSG Execution specific tags
            subfile.write("stream_output = True\n")
            subfile.write("stream_error = True\n\n")
            subfile.write("should_transfer_files = YES\n")
            subfile.write("when_to_transfer_output = ON_SUCCESS\n")
            subfile.write("success_exit_code = 0\n")
            subfile.write("transfer_executable={0}\n".format(str(transfer_executable)))
            subfile.write("transfer_input_files = $(TRANSFER_FILES)\n")

            # Pluralized to support batching output
            subfile.write("transfer_output_files = $(OUTPUT_FILES)\n")
            subfile.write("transfer_output_remaps = $(REMAP_OUTPUT_FILES)\n\n")

            if use_osdf:
                subfile.write("use_oauth_services = scitokens\n")
                # A broken OSDF upload leaves truncated objects and the site's
                # PelicanRetry rerun is then held with "remote object already exists".
                # Remove it so DAGMan RETRY + the PRE cleanup script rerun the node.
                subfile.write(
                    "periodic_remove = (JobStatus == 5) && (HoldReasonCode == 12) && "
                    'regexp("remote object already exists", HoldReason)\n'
                )

        subfile.write("queue 1\n")

    return 0


def write_search_dagfile(file_name, job_name, sub_file_name, job_num, arg_list_string):
    """
    Appends a JOB entry to the Condor DAG file.
    """
    Path(Path(file_name).resolve().parent).mkdir(parents=True, exist_ok=True)
    with open(file_name, "a") as dagfile:
        dagfile.write("JOB {0}_{1} {2}\n".format(job_name, job_num, sub_file_name))
        dagfile.write(
            'VARS {0}_{1} JobID="{1}" {2}'.format(job_name, job_num, arg_list_string)
        )
        dagfile.write("\n")
    return 0
