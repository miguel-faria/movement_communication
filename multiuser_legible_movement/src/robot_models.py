# !/usr/bin/env python
import roboticstoolbox as rtb
import numpy as np


# ABB IRB4600-40 RoboticsToolbox
class IRB4600(rtb.DHRobot):

    def __init__(self, symbolic=False):

        if symbolic:
            import spatialmath.base.symbolic as sym
            zero = sym.zero()
            pi = sym.pi()
        else:
            from math import pi
            zero = 0.0

        deg = pi / 180
        inch = 0.0254

        L = [
            rtb.RevoluteDH(
                d=0.495,
                a=0.175,
                alpha=-pi/2,
                qlim=[-pi, pi]
            ),
            rtb.RevoluteDH(
                d=0,
                a=0.9,
                alpha=0,
                offset=-pi/2,
                qlim=[-pi/2, 150.0*deg]
            ),
            rtb.RevoluteDH(
                d=0,
                a=0.175,
                alpha=-pi/2,
                qlim=[-pi, 75*deg]
            ),
            rtb.RevoluteDH(
                d=0.960,
                a=0,
                alpha=pi/2,
                qlim=[-400*deg, 400*deg]
            ),
            rtb.RevoluteDH(
                d=0,
                a=0,
                alpha=pi/2,
                offset=pi,
                qlim=[-125*deg, 120*deg]
            ),
            rtb.RevoluteDH(
                d=0.135,
                a=0,
                alpha=0,
                qlim=[-400*deg, 400*deg]
            )
        ]

        super().__init__(
            L,
            name='IRB4600-40',
            manufacturer='ABB'
        )

        self.addconfiguration("qz", np.array([0, 0, 0, 0, 0, 0]))
        self.addconfiguration("qd", np.array([0, -90 * deg, 180 * deg, 0, 0, -90 * deg]))
        self.addconfiguration("qr", np.array([90 * deg, -108 * deg, 75 * deg, 0*deg, 135*deg, 0]))


# UR10e RoboticsToolbox
class UR10(rtb.DHRobot):

    def __init__(self, symbolic=False):

        if symbolic:
            import spatialmath.base.symbolic as sym
            zero = sym.zero()
            pi = sym.pi()
        else:
            from math import pi
            zero = 0.0

        deg = pi / 180
        inch = 0.0254

        L = [
            rtb.RevoluteDH(
                d=0.1625,
                a=0,
                alpha=pi/2,
                qlim=[-2*pi, 2*pi]
            ),
            rtb.RevoluteDH(
                d=0,
                a=-0.425,
                alpha=0,
                qlim=[-2*pi, 2*pi]
            ),
            rtb.RevoluteDH(
                d=0,
                a=-0.3922,
                alpha=0,
                qlim=[-pi, pi]
            ),
            rtb.RevoluteDH(
                d=0.1333,
                a=0,
                alpha=pi/2,
                qlim=[-2*pi, 2*pi]
            ),
            rtb.RevoluteDH(
                d=0.0997,
                a=0,
                alpha=-pi/2,
                qlim=[-2*pi, 2*pi]
            ),
            rtb.RevoluteDH(
                d=0.0996,
                a=0,
                alpha=0,
                qlim=[-2*pi, 2*pi]
            )
        ]

        super().__init__(
            L,
            name='UR10',
            manufacturer='Universal Robotics'
        )

        self.addconfiguration(
            "qz", np.array([0, 0, 0, 0, 0, 0]))
        self.addconfiguration(
            "qr", np.array([np.pi, 0, 0, 0, np.pi / 2, 0]))