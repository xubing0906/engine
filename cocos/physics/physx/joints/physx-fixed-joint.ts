/*
 Copyright (c) 2020 Xiamen Yaji Software Co., Ltd.

 https://www.cocos.com/

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated engine source code (the "Software"), a limited,
 worldwide, royalty-free, non-assignable, revocable and non-exclusive license
 to use Cocos Creator solely to develop games on your target platforms. You shall
 not use Cocos Creator software for developing other software or tools that's
 used for developing games. You are not granted to publish, distribute,
 sublicense, and/or sell copies of Cocos Creator.

 The software or tools in this License Agreement are licensed, not sold.
 Xiamen Yaji Software Co., Ltd. reserves all rights not expressly granted to you.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

import { IVec3Like, Vec3, Quat, Mat4 } from '../../../core';
import { FixedConstraint, PhysicsSystem } from '../../framework';
import { IFixedConstraint } from '../../spec/i-physics-constraint';
import { PX, _trans, getTempTransform, _pxtrans } from '../physx-adapter';
import { PxContactPairFlag } from '../physx-enum';
import { PhysXInstance } from '../physx-instance';
import { PhysXRigidBody } from '../physx-rigid-body';
import { PhysXWorld } from '../physx-world';
import { PhysXJoint } from './physx-joint';

export class PhysXFixedJoint extends PhysXJoint implements IFixedConstraint {
    setBreakForce (v: number): void {
        this._breakForce = this.constraint.breakForce;
        this._impl.setBreakForce(this._breakForce, this._breakTorque);
    }

    setBreakTorque (v: number): void {
        this._breakTorque = this.constraint.breakTorque;
        this._impl.setBreakForce(this._breakForce, this._breakTorque);
    }

    get constraint (): FixedConstraint {
        return this._com as FixedConstraint;
    }

    private _breakForce = 0;
    private _breakTorque = 0;

    onComponentSet (): void {
        this._impl = PX.createFixedConstraint(PhysXJoint.tempActor, _pxtrans, null, _pxtrans);
        this.setBreakForce(this.constraint.breakForce);
        this.setBreakTorque(this.constraint.breakTorque);
        this.updateFrame();
    }

    updateFrame () {
        const bodyA = (this._rigidBody.body as PhysXRigidBody).sharedBody;
        const cb = this.constraint.connectedBody;
        const bodyB = cb ? (cb.body as PhysXRigidBody).sharedBody : (PhysicsSystem.instance.physicsWorld as PhysXWorld).getSharedBody(bodyA.node);

        const pos : Vec3 = new Vec3();
        const rot : Quat = new Quat();

        const trans = new Mat4();
        Mat4.fromRT(trans, bodyA.node.worldRotation, bodyA.node.worldPosition);
        Mat4.invert(trans, trans);
        Mat4.getRotation(rot, trans);
        Mat4.getTranslation(pos, trans);
        this._impl.setLocalPose(0, getTempTransform(pos, rot));

        Mat4.fromRT(trans, bodyB.node.worldRotation, bodyB.node.worldPosition);
        Mat4.invert(trans, trans);
        Mat4.getRotation(rot, trans);
        Mat4.getTranslation(pos, trans);
        this._impl.setLocalPose(1, getTempTransform(pos, rot));
    }

    updateScale0 () {
        this.updateFrame();
    }

    updateScale1 () {
        this.updateFrame();
    }
}
