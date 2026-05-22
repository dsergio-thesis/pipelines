
import token
import requests
import time
from urllib.parse import urlencode

from astroquery.utils.tap import TapPlus
import pyvo

from abc import ABC, abstractmethod

class TapClient(ABC):
    def __init__(self, base_url, credentials_file=None, maxrecords=None):
        """
        base_url should be like:
        https://data.lsst.cloud/api/tap
        https://gea.esac.esa.int/tap-server/tap

        """

        self.base_url = base_url.rstrip("/")
        self.credentials_file = credentials_file
        self.maxrecords = maxrecords

    @abstractmethod
    def sync(self, query):
        pass

    @abstractmethod
    def async_submit(self, query):
        pass

    @abstractmethod
    def async_wait(self, job_url, poll_interval=2):
        pass

    @abstractmethod
    def async_result(self, job_url):
        pass

    def query_async(self, query, poll_interval=2):
        job = self.async_submit(query)
        results = job.fetch_result()
        return results.to_table() 

# ============================================================
# requests TAP Client
# ============================================================
class SimpleTAPClient(TapClient):
    def __init__(self, base_url, credentials_file=None):
        super().__init__(base_url, credentials_file=credentials_file)

    def sync(self, query, format="csv"):
        """Run a synchronous TAP query (limited to fast/short jobs)."""
        url = f"{self.base_url}/sync"
        resp = requests.post(url, data={"REQUEST": "doQuery",
                                        "LANG": "ADQL",
                                        "FORMAT": format,
                                        "QUERY": query})
        resp.raise_for_status()
        return resp.text

    def async_submit(self, query, format="csv"):
        """Submit an async TAP query; returns job URL."""
        url = f"{self.base_url}/async"
        resp = requests.post(url, data={"REQUEST": "doQuery",
                                        "LANG": "ADQL",
                                        "FORMAT": format,
                                        "QUERY": query})
        resp.raise_for_status()
        job_url = resp.headers.get("Location")
        return job_url

    def async_wait(self, job_url, poll_interval=2):
        """Wait for an async job to finish."""
        phase_url = job_url + "/phase"
        while True:
            phase = requests.get(phase_url).text.strip()
            if phase in ("COMPLETED", "ERROR", "ABORTED"):
                return phase
            time.sleep(poll_interval)

    def async_result(self, job_url):
        """Download the async result."""
        result_url = job_url + "/results/result"
        resp = requests.get(result_url)
        resp.raise_for_status()
        return resp.text


# ============================================================
# Pyvo TAP Client
# ============================================================
class PyvoTAPClient(TapClient):
    def __init__(self, base_url, credentials_file = None, maxrecords=None):
        super().__init__(base_url, credentials_file=credentials_file, maxrecords=maxrecords)

        if credentials_file is not None:
            self.session = requests.Session()
            # read token from file
            with open(credentials_file, "r") as f:
                token = f.read().strip()
            self.session.headers["Authorization"] = f"Bearer {token}"
        else:
            self.session = None
            token = None

        self.service = pyvo.dal.tap.TAPService(
            base_url,
            session=self.session)
        print("Initialized Pyvo TAP Client.")
        print(f"Token: {token}, self.service: {self.service}")

    def sync(self, query):
        if self.maxrecords is not None:
            return self.service.run_sync(query, maxrec=self.maxrecords)
        else:
            return self.service.run_sync(query)

    def async_submit(self, query):
        print(f"Submitting async TAP query with self.maxrecords: {self.maxrecords}")
        if self.maxrecords is not None:
            job = self.service.submit_job(query, maxrec=self.maxrecords)
        else:
            job = self.service.run_async(query)
        
        # print(f"Job: {job.phase} URL: {job.url}")
        job.run()
        # print(f"Job: {job.phase} URL: {job.url}")
        return job

    def async_wait(self, job, poll_interval=2):
        while job.phase not in ("COMPLETED", "ERROR", "ABORTED"):
            time.sleep(poll_interval)
            # print(f"Job phase: {job.phase}")
        return job.phase

    def async_result(self, job):
        # print(f"Job phase: {job.phase}")
        job = pyvo.dal.tap.AsyncTAPJob(job.url, session=self.session)
        return job.fetch_result()

# ============================================================
# TapPlus TAP Client
# ============================================================
class TapPlusClient(TapClient):

    def __init__(self, base_url, credentials_file=None):
        super().__init__(base_url, credentials_file=credentials_file)
        self.service = TapPlus(url=base_url)

        if (self.credentials_file is not None):
            self.service.login(credentials_file=self.credentials_file)

            print("Connecting to the TapPlus TAP service...")
            if self.service is not None:
                print("Connected to TapPlus TAP service.")
            else:
                print("Failed to connect to TapPlus TAP service.")
        else:
            print("No credentials file provided. Proceeding without authentication.")

        print("Initialized TapPlus Client.")


    def sync(self, query):
        return self.service.run_sync(query)

    def async_submit(self, query):
        return self.service.run_async(query)

    def async_wait(self, job_url, poll_interval=2):
        return self.service.async_wait(job_url, poll_interval=poll_interval)

    def async_result(self, job_url):
        return self.service.async_result(job_url)
