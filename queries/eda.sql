
SELECT *
FROM delta.courier_routing_courier_ml_features_odp.order_level_features AS olf
LIMIT 100
;



SELECT
    olf.order_id                                     AS order_id,
    olf.courier_id                                   AS courier_id,
    olf.country_code                                 AS country_code,
    olf.city_code                                    AS city_code,
    olf.order_activated_local_datetime               AS activation_time,
    olf.courier_transport                            AS transport,
    olf.order_picked_up_local_datetime               AS pickup_time,
    olf.order_delivered_local_datetime               AS delivery_time,
    olf.order_pickup_latitude                        AS pickup_latitude,
    olf.order_pickup_longitude                       AS pickup_longitude,
    olf.order_delivery_latitude                      AS delivery_latitude,
    olf.order_delivery_longitude                     AS delivery_longitude,
    olf.order_arrival_to_delivery_local_datetime     AS delivery_entering_time,
    olf.order_time_zone                              AS time_zone,
    olf.p_creation_date
FROM delta.courier_routing_courier_ml_features_odp.order_level_features AS olf
WHERE order_final_status = 'DeliveredStatus'
    AND order_number_of_assignments = 1
    AND order_bundle_index IS NULL
    AND p_creation_date >= DATE '2024-01-01'
    AND country_code IN ('ES')
/*
    AND {activation_time_constraint(
         start_time=f"DATE('{start_time}')",
         end_time=f"DATE('{end_time}')",
         table=None,
         time_field="p_creation_date",
         add_quotes_around_start_end_time=False
    )}
    AND {city_constraint(
         city_code=self.city_code,
         table=None,
         city_code_field="city_code",
         order_id_field="order_id"
    )}
*/
;