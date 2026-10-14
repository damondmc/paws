"""File paths of the pipeline, derived from the settings."""

from pathlib import Path


def make_dir(filenames):
    """Create the parent directory of each file."""
    for name in filenames:
        Path(name).resolve().parent.mkdir(parents=True, exist_ok=True)


class PathManager:
    """Paths of the DAG, Condor, Weave result and outlier files.

    settings: paws.settings.FixedSettings (or Settings).
    """

    def __init__(self, settings):
        self.settings = settings
        self.home_dir = Path(settings.environment.home_dir)
        self.osdf_dir = Path(settings.environment.osdf_dir)
        self.target_name = settings.target.name
        self.sft_source = settings.data.sft_source
        self.user = settings.environment.user

    @property
    def weave_executable(self):
        return self.settings.environment.executables.weave

    # ---------------------------------------------------------
    # Input Data (SFTs)
    # ---------------------------------------------------------

    def to_osdf_url(self, path):
        """
        Converts a local OSDF mount path (e.g. /osdf/igwn/...) to a pelican
        osdf:// URL by stripping the leading '/osdf' mount prefix (5 chars).
        """
        return "osdf://" + str(path)[5:]

    def sft_file_path(self, freq, detector):
        """Directory of the detector's SFTs of a 1 Hz band (sft_dir in data.yaml is the local OSDF mount path)."""
        return Path(self.settings.data.sft_dir) / detector / str(int(freq))

    def sft_ensemble(self, freq):
        """osdf:// URLs of the SFTs of every detector in the band."""
        return [
            self.to_osdf_url(path)
            for detector in self.settings.data.detectors
            for path in self.sft_file_path(freq, detector).glob("*.sft")
        ]

    # ---------------------------------------------------------
    # Condor & DAG Management
    # ---------------------------------------------------------

    def dag_group_file(self, f_min, f_max, stage):
        """Path to the text file listing all DAGs for a band."""
        filename = f"{self.target_name}_{stage}_{f_min}-{f_max}Hz_dag.txt"
        return self.home_dir / "dagJob" / filename

    def dag_file(self, freq, taskname, stage):
        """Path to the specific .dag file."""
        return (
            self.home_dir
            / "condorFiles"
            / stage
            / self.target_name
            / str(freq)
            / f"{taskname}.dag"
        )

    def condor_sub_file(self, freq, taskname, stage):
        """Path to the .sub file."""
        return (
            self.home_dir
            / "condorFiles"
            / stage
            / self.target_name
            / str(freq)
            / f"{taskname}.sub"
        )

    def condor_record_files(self, freq, taskname, stage):
        """
        Returns a list [Output, Error, Log] for Condor logging.
        Note: These contain $(JobID) or similar Condor variables, so they are returned as strings.
        """
        base_dir = (
            self.home_dir
            / "results"
            / stage
            / self.target_name
            / self.sft_source
            / str(freq)
        )

        # Ensure parent log directories exist immediately (optional but recommended)
        (base_dir / "OUT").mkdir(parents=True, exist_ok=True)
        (base_dir / "ERR").mkdir(parents=True, exist_ok=True)
        (base_dir / "LOG").mkdir(parents=True, exist_ok=True)

        out = base_dir / "OUT" / f"{taskname}.out.$(JobID)"
        err = base_dir / "ERR" / f"{taskname}.err.$(JobID)"
        log = base_dir / "LOG" / f"{taskname}_log.txt.$(JobID)"

        return [str(out), str(err), str(log)]

    # ---------------------------------------------------------
    # Weave Outputs & Analysis
    # ---------------------------------------------------------

    def weave_output_file(self, freq, taskname, job_index, stage):
        """
        Path where Weave writes the resulting FITS file.
        Matches: /osdf/.../o4ab/results/...
        """
        # Logic: OSDFDir + 'o4ab' + results structure
        base = (
            self.osdf_dir
            / "o4ab"
            / "results"
            / stage
            / self.target_name
            / self.sft_source
            / str(freq)
            / "Result"
        )
        return base / f"{taskname}.fts.{job_index}"

    def outlier_file(self, freq, taskname, stage, cluster, location):
        """
        Path of a stage's outlier file (clustered or not).

        location:
          "home"      under home_dir, written by a local collection
          "osdf"      under <osdf_dir>/o4ab, written by an outlier-collection Condor job
                      (OSG jobs can't transfer output straight back to the access point)
          "existing"  the home path if that file exists, else the OSDF path (for readers)
        """
        if location == "existing":
            home_path = self.outlier_file(freq, taskname, stage, cluster, "home")
            return (
                home_path
                if home_path.exists()
                else self.outlier_file(freq, taskname, stage, cluster, "osdf")
            )
        roots = {"home": self.home_dir, "osdf": self.osdf_dir / "o4ab"}
        if location not in roots:
            raise ValueError(
                f"location must be 'home', 'osdf' or 'existing', not {location!r}"
            )
        base = (
            roots[location]
            / "results"
            / stage
            / self.target_name
            / self.sft_source
            / str(freq)
            / "Outliers"
        )
        filename = (
            f"{taskname}_outlier_clustered.fts"
            if cluster
            else f"{taskname}_outlier.fts"
        )
        return base / filename
