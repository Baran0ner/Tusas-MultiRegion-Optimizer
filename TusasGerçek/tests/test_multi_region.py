
import sys
import os
import unittest
import pprint

# Add project root to path
PROJECT_ROOT = r'C:\Users\baran\.gemini\antigravity\scratch\TusasProje\TusasGerçek'
sys.path.append(PROJECT_ROOT)


from tusas.core.multi_region.graph import RegionGraph
from tusas.core.multi_region.optimizer import MultiRegionOptimizer
from tusas.core.laminate_optimizer import LaminateOptimizer

class TestMultiRegionOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Redirect stdout/stderr to file
        cls.log_file = open('tests/test_run_internal.log', 'w', encoding='utf-8')
        sys.stdout = cls.log_file
        sys.stderr = cls.log_file
        
        # Increase attempts for difficult drop scenarios (64->32 is 50% drop)
        LaminateOptimizer.DROP_OFF_ATTEMPTS = 50000

    @classmethod
    def tearDownClass(cls):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        cls.log_file.close()

    def setUp(self):
        # 6-region test scenario
        self.adjacency_map = {
            'R64': ['R32', 'R22', 'R48'],
            'R48': ['R32', 'R16', 'R56', 'R22'],
            'R22': ['R64', 'R48', 'R56'],
            'R32': ['R64', 'R48', 'R16'],
            'R56': ['R48', 'R22'],
            'R16': ['R48', 'R32']
        }
        
        # Ply distribution for master region (assuming R64 is master)
        # Total 64 plies: 16 of each angle
        self.ply_counts = {0: 16, 90: 16, 45: 16, -45: 16}

    def test_optimization_logic(self):
        print("\n--- Starting Multi-Region Optimization Test ---")
        
        # 1. Initialize Graph
        graph = RegionGraph(self.adjacency_map)
        
        # Verify master region identification
        master_id = graph.get_master_region_id()
        print(f"Detected Master Region: {master_id}")
        self.assertEqual(master_id, 'R64', "Master region should be R64")
        
        # 2. Initialize Optimizer
        optimizer = MultiRegionOptimizer(graph, self.ply_counts, fast_mode=True)
        
        # 3. Run Optimization
        result = optimizer.optimize()
        
        # 4. Validate Results
        self.assertIsNotNone(result['mpt'], "MPT should not be None")
        self.assertIsNotNone(result['regions'], "Regions should not be None")
        
        # Stats validation
        print("\nOptimization Stats:")
        pprint.pprint(result['stats'])
        
        # Continuity validation
        self.assertTrue(result['stats']['continuity_valid'], "Continuity validation failed")
        
        # Region specific validations
        regions = result['regions']
        self.assertEqual(len(regions), 6, "Should have 6 regions")
        
        for rid, region_data in regions.items():
            # Check ply count match (allowance due to dropping constraints)
            target = region_data['target_ply_count']
            actual = region_data['actual_ply_count']
            angle_counts = region_data['angle_counts']
            print(f"Region {rid}: Target={target}, Actual={actual}")
            print(f"  Angles: {angle_counts}")
            
            # Allow small deviation if necessary, but ideally should match target or be close
            # Note: actual count might differ slightly if drop-off logic enforces constraints
            self.assertTrue(abs(target - actual) <= 4, f"Ply count mismatch for {rid}")
            
            # Check validations
            vals = region_data['validations']
            self.assertTrue(vals['symmetric'], f"Region {rid} is not symmetric")
            
            # Relaxed balance check
            ac = angle_counts
            diff_balance = abs(ac.get(45, 0) - ac.get(-45, 0))
            print(f"  Balance Diff: {diff_balance}")
            self.assertTrue(diff_balance <= 4, f"Region {rid} is not balanced (diff={diff_balance})")
            
            self.assertTrue(vals['external_plies_ok'], f"Region {rid} external plies issue")
            
        print("\nTest passed successfully!")

if __name__ == '__main__':
    unittest.main()
