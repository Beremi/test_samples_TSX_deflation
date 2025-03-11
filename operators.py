import numpy as np
from scipy import sparse


class BlockMatrixOperator:
    def __init__(self, blocks):
        """
        Initialize with a list of 4 blocks representing a 2x2 block matrix.

        Parameters:
        -----------
        blocks : list
            A list of 4 sparse matrices arranged in row-major order:
            [block00, block01, block10, block11]
        """
        if len(blocks) != 4:
            raise ValueError("Expected 4 blocks for a 2x2 block matrix.")

        self.blocks = blocks

        # Unpack blocks
        block00, block01, block10, block11 = blocks

        # Check compatibility
        if block00.shape[1] != block10.shape[1]:
            raise ValueError(
                f"Blocks (0,0) and (1,0) must have the same number of columns: {
                    block00.shape[1]} vs {
                    block10.shape[1]}")
        if block01.shape[1] != block11.shape[1]:
            raise ValueError(
                f"Blocks (0,1) and (1,1) must have the same number of columns: {
                    block01.shape[1]} vs {
                    block11.shape[1]}")
        if block00.shape[0] != block01.shape[0]:
            raise ValueError(
                f"Blocks (0,0) and (0,1) must have the same number of rows: {
                    block00.shape[0]} vs {
                    block01.shape[0]}")
        if block10.shape[0] != block11.shape[0]:
            raise ValueError(
                f"Blocks (1,0) and (1,1) must have the same number of rows: {
                    block10.shape[0]} vs {
                    block11.shape[0]}")

        # Store dimensions
        self.row_heights = [block00.shape[0], block10.shape[0]]
        self.col_widths = [block00.shape[1], block01.shape[1]]

        # Total dimensions
        self.total_rows = sum(self.row_heights)
        self.total_cols = sum(self.col_widths)
        # Define shape attribute to mimic numpy/scipy matrices
        self.shape = (self.total_rows, self.total_cols)

    def __call__(self, x):
        """
        Multiply the block matrix by vector x.

        Parameters:
        -----------
        x : array-like
            Vector to multiply with.

        Returns:
        --------
        numpy.ndarray
            Result of the matrix-vector multiplication.
        """
        if len(x) != self.total_cols:
            raise ValueError(f"Input vector length ({len(x)}) must match matrix width ({self.total_cols})")

        # Split the input vector
        x1 = x[:self.col_widths[0]]
        x2 = x[self.col_widths[0]:]

        # Perform multiplication for each block and combine results
        block00, block01, block10, block11 = self.blocks
        y1 = block00.dot(x1) + block01.dot(x2)
        y2 = block10.dot(x1) + block11.dot(x2)

        # Combine the results
        return np.concatenate([y1, y2])

    def solve(self, b):
        """
        Solve the system Ax = b where A is this block matrix operator.

        Parameters:
        -----------
        b : array-like
            Right-hand side vector.

        Returns:
        --------
        numpy.ndarray
            Solution of the system.
        """
        if len(b) != self.total_rows:
            raise ValueError(f"Input vector length ({len(b)}) must match matrix height ({self.total_rows})")

        # Construct the full sparse matrix
        block00, block01, block10, block11 = self.blocks

        # Create a sparse block matrix
        top_row = sparse.hstack([block00, block01])
        bottom_row = sparse.hstack([block10, block11])
        full_matrix = sparse.vstack([top_row, bottom_row])

        # Solve the system
        x = sparse.linalg.spsolve(full_matrix, b)

        return x


class BlockTriangularPreconditioner:
    def __init__(self, blocks, verbose=False):
        """
        Initialize with a list of 4 blocks representing a 2x2 block triangular matrix.

        Parameters:
        -----------
        blocks : list
            A list of 4 sparse matrices:
            [block00 (diagonal), block01 (upper), block10 (zero), block11 (diagonal)]
            where block10 must be an empty/zero matrix
        """
        if len(blocks) != 4:
            raise ValueError("Expected 4 blocks for a 2x2 block triangular matrix.")

        self.blocks = blocks

        # Unpack blocks
        block00, block01, block10, block11 = blocks

        # Check that block10 is empty/zero
        if block10.count_nonzero() != 0:
            raise ValueError("Block (1,0) must be an empty/zero matrix for triangular structure")

        # Check compatibility
        if block00.shape[0] != block00.shape[1]:
            raise ValueError(f"Diagonal block (0,0) must be square: {block00.shape}")
        if block11.shape[0] != block11.shape[1]:
            raise ValueError(f"Diagonal block (1,1) must be square: {block11.shape}")
        if block00.shape[0] != block01.shape[0]:
            raise ValueError(f"Block (0,0) rows must match block (0,1) rows: {block00.shape[0]} vs {block01.shape[0]}")
        if block01.shape[1] != block11.shape[1]:
            raise ValueError(
                f"Block (0,1) columns must match block (1,1) columns: {
                    block01.shape[1]} vs {
                    block11.shape[1]}")
        if block10.shape[0] != block11.shape[0]:
            raise ValueError(f"Block (1,0) rows must match block (1,1) rows: {block10.shape[0]} vs {block11.shape[0]}")
        if block10.shape[1] != block00.shape[1]:
            raise ValueError(
                f"Block (1,0) columns must match block (0,0) columns: {
                    block10.shape[1]} vs {
                    block00.shape[1]}")

        # Store dimensions
        self.row_heights = [block00.shape[0], block11.shape[0]]
        self.col_widths = [block00.shape[1], block11.shape[1]]

        # Total dimensions
        self.total_rows = sum(self.row_heights)
        self.total_cols = sum(self.col_widths)
        # Define shape attribute to mimic numpy/scipy matrices
        self.shape = (self.total_rows, self.total_cols)

        self.verbose = verbose

    def __call__(self, b):
        """
        Apply the block triangular preconditioner to vector b.
        This involves solving the system M*x = b where M is the block triangular matrix.

        Parameters:
        -----------
        b : array-like
            Vector to apply the preconditioner to.

        Returns:
        --------
        numpy.ndarray
            Result of applying the preconditioner.
        """
        if len(b) != self.total_rows:
            raise ValueError(f"Input vector length ({len(b)}) must match matrix height ({self.total_rows})")

        if self.verbose:
            print("Applying block triangular preconditioner")

        # Split the input vector
        b1 = b[:self.row_heights[0]]
        b2 = b[self.row_heights[0]:]

        # Unpack blocks
        block00, block01, block10, block11 = self.blocks

        # Solve the system in two steps
        # 1. Solve for x2: block11 * x2 = b2
        x2 = sparse.linalg.spsolve(block11, b2)

        # 2. Solve for x1: block00 * x1 = b1 - block01 * x2
        x1 = sparse.linalg.spsolve(block00, b1 - block01.dot(x2))

        # Combine the results
        return np.concatenate([x1, x2])
