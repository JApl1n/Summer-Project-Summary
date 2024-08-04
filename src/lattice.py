"""Lattice class using ctypes

A short explanation on ctypes:

* a library that provides C data types and allows calling functions in DLLs or
shared libraries.
* to use c/lattice.so, we use CDLL, short for Compile Dynamic Link Libraries
* (so = shared object; a DLL is a shared object that doesn't need to be there
at compilation time; this is not relevant to interpreted languages like Python
?)

"""

import ctypes

import numpy as np

_clattice = ctypes.CDLL("c/lattice.so")  # import c_types
c_int_p = ctypes.POINTER(ctypes.c_int)  # set c_int_p as an int pointer


class Lattice:
    """Lattice class
    (of which particular instances are 'objects')

    # Takes in:
    --------

    :param n_sites: the number of sites on lattice [int]
    :param n_particles: the number of particles on lattice [int]
    :param connectivity: the number of connectivity [int], defaults to 4

    # Contains the following methods:
    --------

    * __init___
    * reset_random_occupancy
    * reset_orientations
    * set_square_connectivity
    * neighbors
    * move
    * positions
    * c_move
    * image

    """

    def __init__(self, n_sites: int, n_particles: int, connectivity=4):  #
        """Initialises object's attributes when initialising/defining an object

        :param self: object's attributes
        :param n_sites: the number of sites on lattice [int]
        :param n_particles: the number of particles on lattice [int]
        :param connectivity: the number of connectivity [int], defaults to 4

        """
        self.n_sites = n_sites
        self.n_particles = n_particles
        self.connectivity = connectivity
        self.occupancy = np.zeros(self.n_sites, dtype=np.int32)
        self.particles = np.ndarray
        self.orientation = np.zeros(self.n_particles, dtype=np.int32)
        self.neighbor_table = np.zeros((self.n_sites, connectivity))
        self.neighbor_table_flat = np.ndarray
        self.n_x = 0
        self.n_y = 0

    def reset_random_occupancy(self):
        """Randomizes the occupancy of the lattice

        :param self: object's attributes

        ------

        # Explanation

        `occupancy` has size `n_sites`, it is either occupied [1] or unoccupied
        [0]

        Explicitly sets `occupancy` to be random:

        * pick a random number from `np.arange(n_sites)` (so `[0..n_sites]`).
        * do this and assign it to every particle and return this as an array
        shape `n_particles`, this effectively gives the location of every
        particle on the lattice
        * for the random sites picked, access
        `occupancy` as the index and set it to 1

        Finally, assign to `particles` the (sorted) index on the lattice where
        it is occupied.


        """
        self.occupancy = np.zeros(self.n_sites, dtype=np.int32)
        self.occupancy[
            np.random.choice(self.n_sites, self.n_particles, replace=False)
        ] = 1
        self.particles = np.where(self.occupancy > 0)[0].astype(dtype=np.int32)

    def reset_orientations(self):
        """Reset orientation of every particle in the lattice

        :param self: object's attributes

        ------

        # Explanation

        Explicitly set `orientation` to be random:

        * Generates an array with random integer between [0,`connectivity`]
        * Sets this to `orientation`

        """
        self.orientation = np.random.randint(
            0, self.connectivity, size=self.n_particles, dtype=np.int32
        )

    def set_square_connectivity(self, n_x, n_y):
        """Forces n_x, n_y dimensions to match number of lattice sites.

        :param self: object's attributes
        :param n_x: number of sites in lattice along x direction
        :param n_y: number of sites in lattice along y direction

        ------

        # Explanation

        Creates (and then flattens) a 2D array of shape [`n_sites`, `connectivity`]

        * This is the list of neighbours that each site has.
        * They are then filled by construct_square_neighbor_table from `lattice.so`
        * Presumably, the neighbour table links each specific lattice site with
        its neighbours for later examination.

        """
        assert n_x * n_y == self.n_sites, "n_x and n_y are incorrect."
        self.n_x = n_x
        self.n_y = n_y
        neighbor_table = np.zeros(
            (self.n_sites, self.connectivity), dtype=np.int32
        ).flatten()

        _clattice._construct_square_neighbor_table(
            neighbor_table.ctypes.data_as(c_int_p), n_x, n_y
        )

        # NOTE: self.neighbor_table_flat = neighbor_table (why not do this instead of
        # reflattening afterwards?

        self.neighbor_table = neighbor_table.reshape(self.n_sites, self.connectivity)
        self.neighbor_table_flat = neighbor_table.flatten()  # and delete this?

    def neighbors(self, site: int):
        """Get the neighbours of a site

        :param self: object's attributes
        :param site: the lattice site index we want the neighbours [int]?
        :returns: the neighbours [np.ndarray]?

        """
        return self.neighbor_table[site]

    def move(self, tumble_probability):
        """Python native implementation to update lattice

        :param self: object's attributes
        :param tumble_probability: the P(tumble) of a particle [float] between [0,1]

        ------

        # Explanation

        To update:

        * pick a random particle from `n_particles`
        * pick its site from `particles`
        * pick its orientation from `orientation`
        * pick a neighbour following the orientation
        * TODO: does some logic I need to figure out

        To tumble:

        * if random integer rolls to be < `tumble_probability`
        * randomizes between [0,`connectivity`] [int]
        * update `orientation` of the particle

        """
        pick = np.random.randint(self.n_particles)
        site = self.particles[pick]
        # attempt = np.random.choice(self.neighbors(site))
        ori = self.orientation[pick]
        attempt = self.neighbors(site)[ori]
        if self.occupancy[attempt] == 0:
            self.occupancy[site] = 0
            self.occupancy[attempt] = 1
            self.particles[pick] = attempt

        if np.random.uniform(0, 1) < tumble_probability:
            self.orientation[pick] = np.random.randint(0, self.connectivity)

    def positions(self) -> tuple:
        """Get the positions of particles as [x,y] coordinates

        :param self: object's attributes
        :returns: [tuple] of form (x,y)

        """
        x, y = np.unravel_index(self.particles, shape=(self.n_x, self.n_y))
        return x, y

    def c_move(self, tumble_probability, speed):
        """Use lattice.so move implementation

        :param self: object's attributes
        :param tumble_probability: the P(tumble) of a particle [float] between [0,1]
        :param speed: the speed of particles [int]

        ------

        TODO: I think it explicitly changes the object. Need to check

        """
        _clattice._move(
            4,
            self.n_particles,
            self.neighbor_table_flat.ctypes.data_as(c_int_p),
            self.orientation.ctypes.data_as(c_int_p),
            self.occupancy.ctypes.data_as(c_int_p),
            self.particles.ctypes.data_as(c_int_p),
            ctypes.c_double(tumble_probability),
            speed,
        )

        assert len(self.particles) == self.n_particles, "not ok"

    def image(self) -> np.ndarray:
        """Define the (x,y) lattice array

        :param self: object's attributes
        :returns: the lattice array [np.ndarray]

        ------

        # Explanation

        Create a matrix of physical size [`n_x`, `n_y`] and fill it as follows:

        * get the coordinates of all particles with positions()
        * set those coordinates to the orientation

        """
        matrix = np.zeros((self.n_x, self.n_y))
        x_pos, y_pos = self.positions()
        matrix[x_pos, y_pos] = self.orientation + 1
        return matrix
