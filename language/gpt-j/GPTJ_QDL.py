
# For QDL
import threading
import requests
from time import sleep
import mlperf_loadgen as lg
import os
import numpy as np
import array

class GPTJ_QDL:
    """QDL acting as a proxy to the SUT.
    This QDL communicates with the SUT via HTTP.
    It uses two endpoints to communicate with the SUT:
    - /predict/ : Send a query to the SUT and get a response.
    - /getname/ : Get the name of the SUT. Send a getname to the SUT and get a response.
    """
    def __init__(self, sut, sut_server_addr: list):
        self.qsl = sut.qsl
        self.sut = sut
        # Construct QDL from the python binding
        self.qdl = lg.ConstructQDL(
            self.issue_query, self.flush_queries, self.client_get_name)
        self.sut_server_addr = sut_server_addr
        self.num_nodes = len(sut_server_addr)

        # For round robin between the SUTs:
        self.next_sut_id = 0
        self.lock = threading.Lock()

    def issue_query(self, query_samples):
        """Process the query to send to the SUT"""
        threading.Thread(target=self.process_query_async,
                         args=[query_samples]).start()

    def flush_queries(self):
        """Flush the queries. Dummy implementation."""
        pass

    def process_query_async(self, query_samples):
        """
        This function is called by the Loadgen in a separate thread.
        It is responsible for
            1. Creating a query for the SUT, by reading the features from the QSL.
            2. Sending the query to the SUT.
            3. Waiting for the response from the SUT.
            4. Deserializing the response.
            5. Calling mlperf_loadgen.QuerySamplesComplete(query_samples, response)
        Args:
            query_samples: A list of QuerySample objects.
        """

        max_num_threads = int(os.environ.get('CM_MAX_NUM_THREADS', os.cpu_count()))

        for i in range(len(query_samples)):
            index = query_samples[i].index
            query = self.sut.data_object.sources[index]
            n = threading.active_count()
            while n >= max_num_threads:
                sleep(0.0001)
                n = threading.active_count()
            threading.Thread(target=self.client_predict_worker,
                         args=[query, query_samples[i].id]).start()
    
    def get_sut_id_round_robin(self):
        """Get the SUT id in round robin."""
        with self.lock:
            res = self.next_sut_id
            self.next_sut_id = (self.next_sut_id + 1) % self.num_nodes
        return res
    
    def client_predict_worker(self, query, query_id):
        """Serialize the query, send it to the SUT in round robin, and return the deserialized response."""
        url = '{}/predict/'.format(self.sut_server_addr[self.get_sut_id_round_robin()])
        responses = []
        print(f"The url:{url} and query:{query}")
        response = requests.post(url, json={'query': query})
        output = response.json()['result']
        response_text = output["response_text"]
        print(f"Response for query: {query} is {response_text}")
        output_batch = np.array(output["pred_output_batch"]).astype(np.int32)
        response_array = array.array("B", output_batch.tobytes())
        bi = response_array.buffer_info()

        responses.append(lg.QuerySampleResponse(query_id, bi[0], bi[1]))
        lg.QuerySamplesComplete(responses)
    
    def client_get_name(self):
        """Get the name of the SUT from ALL the SUTS."""
        if len(self.sut_server_addr) == 1:
            return requests.post(f'{self.sut_server_addr[0]}/getname/').json()['name']
    
        sut_names = [requests.post(f'{addr}/getname/').json()['name'] for addr in self.sut_server_addr]
        return "Multi-node SUT: " + ', '.join(sut_names)

    def __del__(self):
        lg.DestroyQDL(self.qdl)   
