import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time

class BVHParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.skeleton = {}
        self.frames = []
        self.frame_time = None
        self.joint_names = []
        self.num_frames = 0

    def parse(self):
        """Parse the BVH file and extract skeleton hierarchy and motion data."""
        try:
            with open(self.file_path, 'r') as f:
                lines = f.readlines()

            skeleton_lines = []
            motion_lines = []
            motion_start = False

            for line in lines:
                if line.strip().startswith('MOTION'):
                    motion_start = True
                if motion_start:
                    motion_lines.append(line.strip())
                else:
                    skeleton_lines.append(line.strip())

            self._parse_skeleton(skeleton_lines)
            self._parse_motion(motion_lines)

        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
        except Exception as e:
            print(f"Error while parsing BVH: {e}")

    def _parse_skeleton(self, skeleton_lines):
        """Extract skeleton hierarchy and joint information."""
        stack = []
        current_joint = None

        for line in skeleton_lines:
            tokens = line.split()
            if not tokens:
                continue

            if tokens[0] in {"ROOT", "JOINT"}:
                joint_name = tokens[1]
                joint_data = {
                    'name': joint_name,
                    'offset': None,
                    'channels': [],
                    'children': []
                }
                if current_joint:
                    current_joint['children'].append(joint_data)
                stack.append(current_joint)
                current_joint = joint_data

            elif tokens[0] == "OFFSET":
                if current_joint:
                    try:
                        current_joint['offset'] = [float(v) for v in tokens[1:]]
                    except ValueError:
                        print(f"Invalid OFFSET line for {current_joint['name']}, defaulting to [0.0, 0.0, 0.0]")
                        current_joint['offset'] = [0.0, 0.0, 0.0]

            elif tokens[0] == "CHANNELS":
                if current_joint:
                    current_joint['channels'] = tokens[2:]

            elif tokens[0] == "End":
                end_site = {
                    'name': "End Site",
                    'offset': [0.0, 0.0, 0.0],  # Default offset for End Sites
                    'channels': [],
                    'children': []
                }
                if current_joint:
                    current_joint['children'].append(end_site)

            elif tokens[0] == "}":
                if stack:
                    parent = stack.pop()
                    if parent:
                        current_joint = parent
                else:
                    self.skeleton = current_joint  # Finalize skeleton when done

        if not self.skeleton:
            print("Error: Skeleton hierarchy could not be parsed correctly.")
        else:
            print("Skeleton parsed successfully:")
            print(self.skeleton)

    def _parse_motion(self, motion_lines):
        """Extract motion data including frames and frame time."""
        for line in motion_lines:
            if line.startswith("MOTION"):
                continue
            if line.startswith("Frames:"):
                try:
                    self.num_frames = int(line.split()[1])
                except (IndexError, ValueError):
                    print(f"Error parsing 'Frames:' line: {line}")
                    self.num_frames = None
            elif line.startswith("Frame Time:"):
                try:
                    self.frame_time = float(line.split()[2])
                except (IndexError, ValueError):
                    print(f"Error parsing 'Frame Time:' line: {line}")
                    self.frame_time = None
            else:
                try:
                    frame_data = [float(v) for v in line.split()]
                    self.frames.append(frame_data)
                except ValueError:
                    print(f"Skipping invalid motion data line: {line}")

        if self.frames:
            print("First frame data:")
            print(self.frames[0])  # Print the first frame for verification

    def get_statistics(self):
        """Calculate basic motion statistics."""
        if self.num_frames is None or self.frame_time is None:
            print("Error: Missing motion data. Ensure the BVH file is correctly parsed.")
            return {}

        duration = self.num_frames * self.frame_time
        stats = {
            'num_joints': len(self.joint_names),
            'num_frames': self.num_frames,
            'frame_time': self.frame_time,
            'duration': duration
        }
        return stats

    def compare_skeleton(self, other_skeleton):
        """Compare this skeleton with another and identify differences."""
        differences = []
        self_joints = set(self.joint_names)
        other_joints = set(other_skeleton.joint_names)

        missing_in_self = other_joints - self_joints
        missing_in_other = self_joints - other_joints

        if missing_in_self:
            differences.append(f"Joints missing in first file: {missing_in_self}")
        if missing_in_other:
            differences.append(f"Joints missing in second file: {missing_in_other}")

        return differences

class MotionProcessor:
    @staticmethod
    def interpolate_frames(frames1, frames2, blend_factor):
        """Interpolate between two sets of frames."""
        if len(frames1) != len(frames2):
            raise ValueError("Frame sets must have the same length for interpolation.")
        interpolated_frames = []
        for f1, f2 in zip(frames1, frames2):
            interpolated_frame = [(1 - blend_factor) * v1 + blend_factor * v2 for v1, v2 in zip(f1, f2)]
            interpolated_frames.append(interpolated_frame)
        return interpolated_frames

    @staticmethod
    def adjust_framerate(frames, original_fps, target_fps):
        """Adjust the framerate of motion data."""
        ratio = original_fps / target_fps
        adjusted_frames = []
        for i in range(int(len(frames) / ratio)):
            adjusted_frames.append(frames[int(i * ratio)])
        return adjusted_frames

    @staticmethod
    def time_warp(frames, factor):
        """Apply time warping to motion data."""
        warped_frames = []
        for i in np.linspace(0, len(frames) - 1, int(len(frames) * factor)):
            warped_frames.append(frames[int(i)])
        return warped_frames


class MotionVisualizer:
    def __init__(self, skeleton, frames):
        self.skeleton = skeleton
        self.frames = frames
        self.is_paused = False
        self.speed = 1.0

    def plot_skeleton(self, ax, joint_positions):
        """Plot the skeleton structure at a single frame."""
        def plot_joint(joint, parent_pos):
            if not joint or 'offset' not in joint or joint['offset'] is None:
                print(f"Skipping joint with invalid offset: {joint.get('name', 'Unknown')}")
                return

            joint_offset = np.array(joint['offset'])
            joint_pos = parent_pos + joint_offset
            joint_name = joint.get('name', 'Unknown')
            print(f"Plotting joint {joint_name} at position {joint_pos}")

            ax.plot(
                [parent_pos[0], joint_pos[0]],
                [parent_pos[1], joint_pos[1]],
                [parent_pos[2], joint_pos[2]],
                'bo-'
            )

            for child in joint.get('children', []):
                plot_joint(child, joint_pos)

        if not self.skeleton:
            print("Error: Skeleton is empty. Cannot plot.")
            return

        root_pos = np.array([0, 0, 0])
        plot_joint(self.skeleton, root_pos)


    def visualize(self):
        """Visualize the skeleton in motion."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        for frame in self.frames:
            if self.is_paused:
                continue

            ax.clear()
            self.plot_skeleton(ax, frame)
            plt.pause(0.05 / self.speed)

        plt.show()

# Main Execution
bvh1_path = "C:\\Users\\BEN\\Downloads\\New folder\\B.bvh"
bvh2_path = "C:\\Users\\BEN\\Downloads\\New folder\\C.bvh"

parser1 = BVHParser(bvh1_path)
parser1.parse()

parser2 = BVHParser(bvh2_path)
parser2.parse()

# Display statistics and differences
stats1 = parser1.get_statistics()
stats2 = parser2.get_statistics()

differences = parser1.compare_skeleton(parser2)

print("File 1 Statistics:", stats1)
print("File 2 Statistics:", stats2)
print("Skeleton Differences:", differences)

# Visualize motion
visualizer = MotionVisualizer(parser1.skeleton, parser1.frames)
visualizer.visualize()
