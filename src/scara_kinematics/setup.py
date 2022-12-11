from setuptools import setup

package_name = 'scara_kinematics'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ed',
    maintainer_email='ed@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "forward_kin_controller = scara_kinematics.forward_kin_controller:main",
            "inverse_kin_client = scara_kinematics.inverse_kin_client:main",
            "inverse_kin_service = scara_kinematics.inverse_kin_service:main",
            "pd_controller = scara_kinematics.pd_controller:main",
            "velocity_converter = scara_kinematics.velocity_converter:main"
        ],
    },
)

