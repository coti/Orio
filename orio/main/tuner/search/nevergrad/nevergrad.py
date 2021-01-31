import time
import orio.main.tuner.search.search
from orio.main.util.globals import *
import numpy as np
import nevergrad as ng

class Nevergrad(orio.main.tuner.search.search.Search):

    def __init__(self, params, **kwargs):
        orio.main.tuner.search.search.Search.__init__(self, params)

        self.__readAlgoArgs()

        # Parameter space

        hp = {}
        for pn,pr in zip( self.axis_names, self.axis_val_ranges ):
            p0 = ng.p.Choice( pr )
            hp[pn] = p0

        self.ng_params = ng.p.Dict( **hp )

        # Create the optimizer
        # Budget = nb of runs
        self.optimizer = ng.optimizers.OnePlusOne( parametrization = self.ng_params, budget = self.total_runs )

        # Constraints
        # A constraint is called "cheap" when ng does not try to reduce the number
        # of calls to such constraints: it repeats mutations until it gets a satisfiable point.
        
        #info( "Constraints: " + str( self.constraint ) )
        #for cons in self.constraints:
        self.optimizer.parametrization.register_cheap_constraint( self.isValid )
        



    def searchBestCoord(self, startCoord=None):
        start_time = time.time()

        #        while not ((time.time()-start_time) > self.time_limit > 0):
        recommendation = self.optimizer.minimize( self.myPerfCost )
        point = self.pointToCoord( recommendation.value )
        search_time = time.time()
        runs = self.optimizer.num_ask

        # Can we get this from Nevergrad?
        fitness = np.average( self.myPerfCost( recommendation.value ) )
        return point, fitness, search_time, runs

    def myPerfCost( self, point ):
        #print( "point: " + str( point ) )
        if not self.isValid( point ):
            return float('inf')
        coord = self.pointToCoord( point )
        return self.getPerfCost( coord )
    
    def isValid( self, point ):
        return eval( self.constraint, point )

    def __readAlgoArgs( self ):
        return

    def pointToCoord( self, point ):
        coord = []
        for idx,name in enumerate( self.axis_names ):
            value = point[ name ]
            coord.append( self.axis_val_ranges[idx].index( value ) )
        return coord
    
