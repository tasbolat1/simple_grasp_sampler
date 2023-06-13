#!/usr/bin/env python3

import numpy as np
import cv2
import rospy
from sensor_msgs.msg import PointCloud2
import sys
import pcl
import pcl.pcl_visualization
from utils.utils_pcl import *
import ros_numpy
import open3d as o3d

class GraspGenerator():
    def __init__(self):

        self._pc_pub = rospy.Publisher(
            "/grasp_generator/pc",
            PointCloud2,
            queue_size=1
        )
        self._pc_sub = rospy.Subscriber(
            "/camera/depth/color/points",
            PointCloud2,
            self.callback_pc,
            queue_size=1,
        )

        self._grasp_pub = rospy.Publisher(
            "/grasp_generator/grasp_pc",
            PointCloud2,
            queue_size=1
        )

        self.latest_pc = None

        self.workspace_xmin = rospy.get_param('/grasp_generator/pc_filter/passthrough/x_min')
        self.workspace_xmax = rospy.get_param('/grasp_generator/pc_filter/passthrough/x_max')
        self.workspace_ymin = rospy.get_param('/grasp_generator/pc_filter/passthrough/y_min')
        self.workspace_ymax = rospy.get_param('/grasp_generator/pc_filter/passthrough/y_max')
        self.workspace_zmin = rospy.get_param('/grasp_generator/pc_filter/passthrough/z_min')
        self.workspace_zmax = rospy.get_param('/grasp_generator/pc_filter/passthrough/z_max')
        self.table_height = rospy.get_param('/grasp_generator/table_height')

    def run(self):
        while not rospy.is_shutdown():
            if self.latest_pc is None:
                continue

            # if not (self.pc is None):
                # visual = pcl.pcl_visualization.CloudViewing()

                # PointXYZ
                # visual.ShowColorCloud(self.pc, b'cloud')

            # publish 

            pc = self.filter_pcl(self.latest_pc)
            bb_points, final_points = self.create_bb(pc)
            self.generate(bb_points, final_points)
            # self.test(self.latest_pc)
            self._pc_pub.publish(pcl_to_ros(pc, frame_id="camera_color_optical_frame"))
            rospy.sleep(0.01)
            

    def callback_pc(self, msg):
        self.latest_pc = ros_to_pcl2(msg)

        # self.test(self.latest_pc)

        # print(self.latest_pc.to_array()[:,3])
        # print(msg.data)
        # self.pc = ros_to_pcl2(msg)
        # print(pc)
        # self.pc = self.filter_pcl(self.pc)
        # self._pc_pub.publish(pcl_to_ros(self.pc, frame_id=msg.header.frame_id))
        

    def generate(self, grasp_points, bb_box):
        grasp_points = np.concatenate([grasp_points, bb_box], axis=0)
        
        colors = np.ones([grasp_points.shape[0],1])*0.5
        grasp_points = np.append(grasp_points, colors, axis=1)#.astype('float32')
        # grasp_points[:,2] =  grasp_points[:,2] - 0.1
        # print(grasp_points.shape)
        # print(colors.shape)
        # print(colors)

        p = pcl.PointCloud_PointXYZRGB()
        p.from_list(grasp_points)
        self._grasp_pub.publish(pcl_to_ros(p, frame_id="camera_color_optical_frame"))
        

    def create_bb(self, pc, add_table_points=True):
        pc_array = pc.to_array()[:,:3]

        if add_table_points:
            [x_add, y_add, z_add]  = pc_array.mean(axis=0)
            z_add = self.table_height
            addable = np.array([x_add, y_add, z_add]).reshape([1,3])
            pc_array = np.append(pc_array, addable, axis=0)

        # print(pc_array.shape)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pc_array)
        # print(pcd)
        o3d_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pc_array))
        bb_points = np.asarray(o3d_bbox.get_box_points())

        # # ignore
        # colors = np.ones([bb_points.shape[0],1])*0.5
        # grasp_points = np.append(bb_points, colors, axis=1)#.astype('float32')
        # # grasp_points[:,2] =  grasp_points[:,2] - 0.1

        # p = pcl.PointCloud_PointXYZRGB()
        # p.from_list(grasp_points)
        # self._grasp_pub.publish(pcl_to_ros(p, frame_id="camera_color_optical_frame"))

        # ignore

        print(bb_points)
        # print(o3d_bbox.center)
        # print(np.mean(bb_points[3:7,:], axis=0))
        [x_c, y_c, z_c] = bb_points[3:7,:].mean(axis=0)
        len1 = np.linalg.norm(bb_points[3,:]-bb_points[5,:])
        len2 = np.linalg.norm(bb_points[4,:]-bb_points[6,:])

        if len1 >= len2:
            [x1,y1,_] = (bb_points[3,:]+bb_points[5,:])/2
            [x2,y2,_] = (bb_points[4,:]+bb_points[6,:])/2
            
        else:
            [x1,y1,_] = (bb_points[3,:]+bb_points[6,:])/2
            [x2,y2,_] = (bb_points[4,:]+bb_points[5,:])/2
        z1 = z_c
        z2 = z_c

        final_res = np.array([[x_c, y_c, z_c], [x1,y1,z1], [x2,y2,z2]])




        return bb_points, final_res


        return o3d_bbox
        # print(pcd)
        # color = ros_numpy.point_cloud2.split_rgb_field(pc)
        # print(pc_array.shape)
        # print(pc_array.shape)
        # print(pc_array[:,3])




    def filter_pcl(self, pc):
        '''
        Inputs: pc - pointcloud as XYZRGB format using pcl
        '''
        passthrough = pc.make_passthrough_filter()
        passthrough.set_filter_field_name("x")
        passthrough.set_filter_limits(self.workspace_xmin, self.workspace_xmax)
        pc = passthrough.filter()

        passthrough = pc.make_passthrough_filter()
        passthrough.set_filter_field_name("y")
        passthrough.set_filter_limits(self.workspace_ymin, self.workspace_ymax)
        pc = passthrough.filter()	

        passthrough = pc.make_passthrough_filter()
        passthrough.set_filter_field_name("z")
        passthrough.set_filter_limits(self.workspace_zmin, self.workspace_zmax)
        pc = passthrough.filter()

        seg = pc.make_segmenter()
        max_distance = 0.015
        seg.set_distance_threshold(max_distance)
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        inliers, coefficients = seg.segment()
        extracted_outliers = pc.extract(inliers, negative=True)
        # extracted_outliers = XYZRGB_to_XYZ(extracted_outliers)
        #print('Segmentation removal: ', extracted_outliers.size)

        return extracted_outliers

def main(argv):
    rospy.init_node("grasp_generator")
    node = GraspGenerator()
    node.run()

    # rospy.spin()


if __name__ == "__main__":
    main(sys.argv)

