const fs = require('fs');

async function fetchCostcoOrders() {
  try {
    console.log('Sending request to Costco GraphQL API...');
    
    const queryVars = {
      "pageNumber": 1,
      "pageSize": 10,
      "startDate": "2025-3-01",
      "endDate": "2025-5-31",
      "warehouseNumber": "847"
    };

    const requestBody = {
        query: `query 
        getOnlineOrders($startDate: String!, $endDate: String!, $pageNumber: Int , $pageSize: Int, $warehouseNumber: String!) {
            getOnlineOrders(startDate: $startDate, endDate: $endDate, pageNumber: $pageNumber, pageSize: $pageSize, warehouseNumber: $warehouseNumber) {
              pageNumber
              pageSize
              totalNumberOfRecords
              bcOrders {
                orderHeaderId
                orderPlacedDate : orderedDate
                orderNumber : sourceOrderNumber 
                orderTotal
                warehouseNumber
                status
                emailAddress
                orderCancelAllowed
                orderPaymentFailed : orderPaymentEditAllowed
                orderReturnAllowed
                orderLineItems {
                  orderLineItemCancelAllowed
                  orderLineItemId
                  orderReturnAllowed
                  itemId
                  itemNumber
                  itemTypeId
                  lineNumber
                  itemDescription
                  deliveryDate
                  warehouseNumber
                  status
                  orderStatus
                  parentOrderLineItemId
                  isFSAEligible
                  shippingType
                  shippingTimeFrame
                  isShipToWarehouse
                  carrierItemCategory
                  carrierContactPhone
                  programTypeId
                  isBuyAgainEligible
                  scheduledDeliveryDate
                  scheduledDeliveryDateEnd
                  configuredItemData
                  price
                }
              }
            }
        }`,
        variables: queryVars
      };

    const getOrderDetailsQuery = {
        query: `query getOrderDetails($orderNumbers: [String]) {
                    getOrderDetails(orderNumbers: $orderNumbers) {
                        warehouseNumber
                        orderNumber: sourceOrderNumber
                        orderPlacedDate: orderedDate
                        status
                        locale
                        orderReturnAllowed
                        shopCardAppliedAmount
                        giftOfMembershipAppliedAmount
                        orderCancelAllowed
                        orderPaymentFailed: orderPaymentEditAllowed
                        orderShippingEditAllowed
                        merchandiseTotal
                        retailDeliveryFee
                        shippingAndHandling
                        grocerySurcharge
                        frozenSurchargeFee
                        uSTaxTotal1
                        foreignTaxTotal1
                        foreignTaxTotal2
                        foreignTaxTotal3
                        foreignTaxTotal4
                        orderTotal
                        firstName
                        lastName
                        line1
                        line2
                        line3
                        city
                        state
                        postalCode
                        countryCode
                        companyName
                        emailAddress
                        phoneNumber
                        membershipNumber
                        nonMemberSurchargeAmount
                        discountAmount

                        retailDeliveryFees {
                        key
                        value
                        }
                        orderPayment {
                        paymentType
                        totalCharged
                        cardExpireMonth
                        cardExpireYear
                        nameOnCard
                        cardNumber
                        isGOMPayment
                        storedValueBucket
                        }
                        shipToAddress: orderShipTos {
                        referenceNumber
                        firstName
                        lastName
                        line1
                        line2
                        line3
                        city
                        state
                        postalCode
                        countryCode
                        companyName
                        emailAddress
                        phoneNumber: contactPhone
                        isShipToWarehouse
                        addressWarehouseName
                        giftMessage
                        giftToFirstName
                        giftToLastName
                        giftFromName
                        orderLineItems {
                            shipToWarehousePackageStatus
                            orderStatus
                            orderNumber
                            orderedDate
                            itemTypeId
                            isFeeItem
                            orderLineItemCancelAllowed
                            estimatedDeliveryDate
                            supplierAvailabilityDate
                            fulfilledBy
                            itemNumber
                            itemDescription: sourceItemDescription
                            price: unitPrice
                            quantity: orderedTotalQuantity
                            merchandiseTotalAmount
                            lineItemId
                            sourceLineItemId
                            parentOrderLineItemId
                            itemId
                            isBuyAgainEligible
                            sequenceNumber: sourceSequenceNumber
                            parentOrderNumber
                            lineNumber
                            itemTypeId
                            replaceStatus
                            returnType
                            itemType
                            programType
                            minOrderDate
                            maxOrderDate
                            fSADescription
                            odsJobId
                            orderedShipMethodDescription
                            shippingChargeAmount
                            preferredArrivalDate
                            requestedDeliveryDate
                            returnStatus
                            productSerialNumber
                            configuredItemData
                            orderedShipMethod
                            isRescheduleEligible
                            deliveryReschedulingSite
                            scheduledDeliveryDate
                            scheduledDeliveryDateEnd
                            limitedReturnPolicyRule
                            isLimitedReturnPolicyExceeded
                            itemWeight
                            itemGroupNumber
                            isPerishable
                            carrierItemCategory
                            carrierContactPhone
                            isUPSMILabelEligible
                            parentLineNumber
                            isExchangeBlock
                            shipToAddressReferenceNumber
                            isVerificationRequired
                            isDept24
                            returnableQuantity
                            totalReturnedQuantity
                            exchangeOrderNumber
                            returnType
                            isGiftMessageSupported
                            isReturnCalendarEligible
                            programTypeId
                            inventoryWarehouseId
                            bundleParentNumber
                            freightSavings
                            freightAdditionalSavings
                            configuredItemData
                            itemStatus {
                            orderPlaced {
                                quantity
                                transactionDate
                                orderLineItemId
                                lineItemStatusId
                                orderLineItemCancelAllowed
                                orderLineItemReturnAllowed
                            }
                            readyForPickup {
                                quantity
                                transactionDate
                                orderLineItemId
                                lineItemStatusId
                                orderLineItemCancelAllowed
                                orderLineItemReturnAllowed
                            }
                            shipped {
                                quantity
                                transactionDate
                                orderLineItemId
                                lineItemStatusId
                                orderLineItemCancelAllowed
                                orderLineItemReturnAllowed
                            }
                            cancelled {
                                quantity
                                transactionDate
                                orderLineItemId
                                lineItemStatusId
                                orderLineItemCancelAllowed
                                orderLineItemReturnAllowed
                            }
                            returned {
                                quantity
                                transactionDate
                                orderLineItemId
                                lineItemStatusId
                                orderLineItemCancelAllowed
                                orderLineItemReturnAllowed
                            }
                            delivered {
                                quantity
                                transactionDate
                                orderLineItemId
                                lineItemStatusId
                                orderLineItemCancelAllowed
                                orderLineItemReturnAllowed
                            }
                            cancellationRequested {
                                quantity
                                transactionDate
                                orderLineItemId
                                lineItemStatusId
                                orderLineItemCancelAllowed
                                orderLineItemReturnAllowed
                            }
                            }
                            shipment {
                            lineNumber
                            orderNumber
                            packageNumber
                            trackingNumber
                            pickUpCompletedDate
                            pickUpReadyDate
                            carrierName
                            trackingSiteUrl
                            shippedDate
                            estimatedArrivalDate
                            deliveredDate
                            isDeliveryDelayed
                            isEstimatedArrivalDateEligible
                            reasonCode
                            trackingEvent {
                                event
                                carrierName
                                eventDate
                                estimatedDeliveryDate
                                scheduledDeliveryDate
                                trackingNumber
                            }
                            }
                        }
                        }
                    }
                    }`,
        variables: {
            "orderNumbers": [
                "1184243605"
            ]
        }
    };

    const introspectionQuery = {
        query: `query {
            __schema {
                types {
                    name
                }
            }
        }`
    };

    const response = await fetch("https://ecom-api.costco.com/ebusiness/order/v1/orders/graphql", {
      "headers": {
        "accept": "*/*",
        "accept-language": "en-IN,en-US;q=0.9,en-GB;q=0.8,en;q=0.7",
        "client-identifier": "",
        "content-type": "application/json-patch+json",
        "costco-x-authorization": "",
        "costco-x-wcs-clientid": "",
        "costco.env": "ecom",
        "costco.service": "restOrders",
        "sec-ch-ua": "\"Google Chrome\";v=\"135\", \"Not-A.Brand\";v=\"8\", \"Chromium\";v=\"135\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Linux\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "Referer": "https://www.costco.com/",
        "Referrer-Policy": "strict-origin-when-cross-origin"
      },
      "body": JSON.stringify(requestBody),
      "method": "POST"
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    console.log('Response status:', response.status);
    // console.log('Response headers:', Object.fromEntries(response.headers));

    const data = await response.json();
    const filename = `costco_orders_${Date()}.json`;
    fs.writeFileSync(filename, JSON.stringify(data, null, 2));
    console.log(`Response data written to ${filename}`);
    
    return data;
  } catch (error) {
    console.error('Error fetching Costco orders:', error);
    throw error;
  }
}

// Execute the function
fetchCostcoOrders()
  .then(data => {
    console.log('Successfully fetched orders');
  })
  .catch(error => {
    console.error('Failed to fetch orders:', error);
    process.exit(1);
  });