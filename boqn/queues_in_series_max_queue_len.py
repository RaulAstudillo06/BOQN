import numpy as np
import matplotlib.pyplot as plt
import torch

# This is the penalty that we pay for increasing the max queue size
QUEUE_SIZE_PENALTY = 0.1

class queues_in_series:

    def __init__(self, nqueues, arrival_rate=1., seed=None, loglevel=0):
        # nqueues: number of queues
        # arrival_rate: a scalar giving the arrival rate of jobs at the first queue
        # loglevel: an integer (0, 1 or 2) telling the level of logging to do.
        # 0 means no logging
        # 1 means to create completion and arrival records but don't print anything.
        # 2 means to additionally print verbose information


        self.nqueues = nqueues
        self.max_server_capacity = 1.2 * nqueues
        self.arrival_rate = arrival_rate

        # This will keep track of the jobs in service in each queue
        self.qlen = [0] * self.nqueues

        # Initialize system clock to 0
        self.t = 0

        # Track how many jobs have completed service at each queue
        self.ncompletions = [0] * self.nqueues
        
        # Set random seed
        self.seed = seed
        self.random_state = np.random.RandomState(seed)

        self.loglevel = loglevel

        # Create 2-dimensional arrays tracking when event arrivals and completions happenend.
        # completions[q] is an array giving the job service completion times, in order.
        # It is initialized to an empty array.
        # Similarly, arrivals[q] gives the job arrival times and is initially empty.
        # This can slow down long simulations, and isn't needed for the current functionality,
        # so we give an option to turn this off.  Later, this (or something like it) would be needed if we want to
        # output statistics about the interarrival times at each queue other than the throughput
        if self.loglevel >= 1:
            self.completions = [[0.]*0 for q in range(self.nqueues)]
            self.arrivals = [[0.]*0 for q in range(self.nqueues)]


    def step(self):
        # Step forward one unit in time

        # The different events that can happen next are:
        # a non-empty queue can have the job in service complete
        # an arrival

        # First we calculate the rate at which each event is happening, and store it in a vector r
        # The event of completing service in queue q is stored in r[q]
        # The event of an arrival is stored in r[nqueues]
        r = np.zeros(self.nqueues+1)

        for q in range(self.nqueues):
            if self.qlen[q] > 0:
                r[q] = self.service_rate[q]

        r[self.nqueues] = self.arrival_rate

        # Simulate the time until the next event, which is an exponential with rate sum(r)
        elapsed = self.random_state.exponential(1./sum(r))

        # Simulate the identity of the next event
        event = self.random_state.choice(range(self.nqueues+1), p=r/sum(r))

        if self.loglevel >= 2:
            print('time={} qlen={} p={} r={} event={} mean_step={}'.format(self.t,self.qlen,r/sum(r),r,event,1./sum(r)))

        # Update the clock
        self.t = self.t+elapsed

        # Update the system state
        if event<self.nqueues: # Service completion
            assert(self.qlen[event]>0)
            if self.loglevel >= 1:
                assert(len(self.completions[event])==self.ncompletions[event])
            self.ncompletions[event] += 1
            if self.loglevel >= 1:
                self.completions[event].append(self.t)
            self.qlen[event] -= 1 # remove the job from the current q

            if event < self.nqueues-1: # If this wasn't the last queue
                if self.qlen[event+1] < self.max_queue_len[event+1]: # if there is space in the next queue
                    self.qlen[event+1] += 1 # add the job to the next queue
                if self.loglevel >= 1:
                    self.arrivals[event+1].append(self.t)

        else: # Arrival
            if self.loglevel >= 1:
                self.arrivals[0].append(self.t)
            if self.qlen[0] < self.max_queue_len[0]:  # if there is space in this queue
                self.qlen[0] += 1

    def simulate(self, maxT):
        # Run a simulation until we hit a maximum time
        while self.t < maxT:
            self.step()

    def throughput(self):
        # Return the throughput at each queue
        return self.ncompletions

    def max_queue_size_penalty(self):
        return QUEUE_SIZE_PENALTY * np.sum(self.max_queue_len)

    def objective(self):
        # Objective function
        return self.throughput() - self.max_queue_size_penalty()

    def inservice(self):
        # Return the number of items of work that are currently in service
        return self.qlen

    def reset(self):
        # Set the simulation to time 0 and reset all other state to allow re-running the simulation
        # This will keep track of the jobs in service in each queue
        self.qlen = [0] * self.nqueues
        self.t = 0
        self.ncompletions = [0] * self.nqueues
        if self.loglevel >= 1:
            self.completions = [[0.] * 0 for q in range(self.nqueues)]
            self.arrivals = [[0.] * 0 for q in range(self.nqueues)]

    def simulate_several(self, nreps=500, maxT=10):
        # Run the simulation several times and compute sample mean and stderr for the output of each queue
        self.random_state = np.random.RandomState(self.seed)
        # For each queue, we'll maintain a list of outputs
        outputs = np.empty((0, self.nqueues))

        for i in range(nreps):
            self.reset()
            self.simulate(maxT)
            # Store the outputs so that output[i,q] stores the throughput for simulation i and queue q
            outputs = np.append(outputs,[self.objective()],axis=0)

        # Calculate mean and standard deviation for each queue
        avg = [np.mean(outputs[:,q]) for q in range(self.nqueues) ]
        stderr = [np.std(outputs[:,q])/np.sqrt(nreps) for q in range(self.nqueues) ]

        if self.loglevel >= 1:
            for q in range(self.nqueues):
                print('queue {} objective avg={:.2f} stderr={:.2f}'.format(q, avg[q], stderr[q]))

        return avg, stderr
    
    def evaluate(self, service_rates_tensor, max_queue_len_tensor):
        service_rates_tensor_copy = (1.2 * self.nqueues) * service_rates_tensor.clone()
        max_queue_len_tensor_copy = (1.2 * self.nqueues) * max_queue_len_tensor.clone()
        input_shape = service_rates_tensor_copy.shape
        output = torch.empty(input_shape[:-1] + torch.Size([self.nqueues, 2]))
        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                self.service_rate = list(service_rates_tensor_copy[i, j, :])
                self.max_queue_len = list(max_queue_len_tensor_copy[i, j, :])
                if len(self.service_rate) == self.nqueues - 1:
                    print(error)#self.service_rate.append(self.nqueues * 1.2 - sum(self.service_rate))
                avg, stderr = self.simulate_several()
                output[i, j, :, 0] = torch.tensor(avg)
                output[i, j, :, 1] = torch.tensor(stderr)
        output = output.double()
        return output

    @staticmethod
    def test1():
        n_servers = [1,1]
        # Since the arrival rate is less than all of the service rates, we should see the throughput close to the
        # total number of arrivals
        arrival_rate = 0.8 # If this is closer to 1, the last check takes more runtime to make actual close to theory
        T = 100000 # Run for a long time to make the answers accurate
        q = queues_in_series(n_servers,arrival_rate,1)
        q.simulate(T)

        # Simple check --- the number of completions should be equal to the length of the list of completion times
        for i in range(q.nqueues):
            assert(len(q.completions[i]) == q.throughput()[i])

        # The number of arrivals at the first queue should be roughly T*arrival rate
        actual=len(q.arrivals[0])
        theory = T*arrival_rate
        diff = 100*np.abs((actual - theory)/theory)
        print('# arrivals at queue 0: actual={} theory={} % difference={}%'.format(actual,theory,diff))
        assert(diff<5)

        theory = arrival_rate * T
        actual = q.throughput()[len(n_servers)-1]
        diff = 100*np.abs((actual - theory)/theory)
        print('Last queue throughput: actual={} theory={} difference={}%'.format(actual,theory,diff))
        assert(diff<5)

        # The average waiting time in the first queue is 1/(service_rate-arrival_rate)
        # See "Response Time", https://en.wikipedia.org/wiki/M/M/1_queue
        mu = n_servers[0] # service rate at first queue
        assert(arrival_rate < mu) # This check only works if the service rate is faster than the arrival rate
        theory = 1/(mu-arrival_rate)
        actual = 0
        for i in range(len(q.arrivals[0])):
            if i < len(q.completions[0]):
                # This job completed service
                completion_time = q.completions[0][i]
            else:
                # This job is still in queue at completion time
                completion_time = T

            wait = completion_time - q.arrivals[0][i] # Time this job waited
            actual += wait
        actual = actual / len(q.arrivals[0]) # avg wait time per job
        diff = 100*np.abs((actual - theory)/theory)
        print('Waiting time at queue 0: actual={} theory={} difference={}%'.format(actual,theory,diff))

        print('throughput {}'.format(q.throughput()))
        print('in service {}'.format(q.inservice()))
        assert(diff<20)


    def test2():
        n_servers = [1,1,1]
        arrival_rate = 1
        T = 10
        q = queues_in_series(n_servers,arrival_rate,1)
        m,s = q.simulate_several(5,T)
        # we should see that the throughput is larger for the earlier queues,
        # since T might not have been long enough to let work get all the way through the queue,
        # and because work may build up at one queue
        assert(m[0] >= m[1])
        assert(m[1] >= m[2])

    def test3():
        arrival_rate = 1
        T = 10
        nreps = 100

        x = [ i/5. for i in range(10)]
        nq = 3

        value = np.zeros((nq, len(x)))
        stderr = np.zeros((nq, len(x)))

        for ix in range(len(x)):
            n_servers = [1,x[ix],1]
            assert(nq == len(n_servers))
            q = queues_in_series(n_servers,arrival_rate,0)
            ret_avg,ret_stderr = q.simulate_several(nreps,T)
            for q in range(nq):
                value[q,ix] = ret_avg[q]
                stderr[q,ix] = ret_stderr[q]

        for q in range(nq):
            plt.errorbar(x, value[q,:], 2*stderr[q,:])
        plt.legend(['q=0', 'q=1', 'q=2'])
        plt.xlabel('x (number of servers at q=1)')
        plt.ylabel('Throughput est +/- 2*stderr ')
        plt.title('arrival_rate={}, T={}, n_servers = [1, x, 1]'.format(arrival_rate,T))
        plt.show()
