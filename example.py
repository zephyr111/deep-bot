import pickle
import collections

# Current state and next actions
TraceItem = collections.namedtuple('TraceItem', ['map', 'bombs', 'actions', 'points'])

trace = list()
# trace.append(TraceItem(...))

with open('trace.pkl', 'wb+') as traceFile:
    pickle.dump(trace, traceFile)
